import os
import types
from collections import defaultdict
from tqdm.auto import tqdm 
from typing import List, Dict, Union, Callable
import logging
from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from ..Functions import *
from ..Modules import *
from .GPTQLinearModuleQuantizer import *
from ...Quantizer import Quantizer
from GPTFast.Helpers import *
from GPTFast.Kernels import pack_int4_weights

logger = getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GPTQModelQuantizerNew(Quantizer):
    def __init__(self, model_name:str, calibration_data_fn:Callable[..., Dict[str, Union[torch.LongTensor, list[int]]]], quantize_config:dict, device:torch.device = "cpu"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(torch.bfloat16)
        self.config = self.model.config
        self.calibration_data_fn = types.MethodType(calibration_data_fn, self)
        self.quantize_config = quantize_config
        self.device = device
        self.model = self.model.to(self.device)
        model_suffix = self.model_name.split("/")[-1]
        self.quantized_state_dict_path = f"{model_suffix}-gptq.pth" if self.quantize_config["save_quantized_state_dict"] else None
        self._quantized = False
        self._quantized_state_dict = {}

    @property
    def quantized_state_dict(self):
        return self._quantized_state_dict

    def skip_layer_func(self, name:str, linear_module:Union[nn.Linear, transformers.pytorch_utils.Conv1D]) -> bool:
        linear_weight = linear_module.weight
        return (name in self.quantize_config["skipped_layers"]) or (not check_linear_int4_k(linear_weight.shape[-1], self.quantize_config["groupsize"]))
    
    def replace_all_linear_layers(self):
        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Linear, transformers.pytorch_utils.Conv1D)) and (not self.skip_layer_func(name, child)):
                    if isinstance(child, transformers.pytorch_utils.Conv1D):
                        in_features, out_features = child.weight.shape[0], child.weight.shape[-1]
                    else:
                        in_features, out_features = child.weight.shape[-1], child.weight.shape[0]

                    groupsize = self.quantize_config["groupsize"]
                    inner_k_tiles = groupsize // 16

                    if check_linear_int4_k(in_features, groupsize, inner_k_tiles) or self.quantize_config.get("padding_allowed", True):
                        new_module = WeightOnlyInt4Linear(
                            in_features, out_features, bias=False,
                            groupsize=groupsize, inner_k_tiles=inner_k_tiles
                        )
                        setattr(module, name, new_module)
                else:
                    replace_linear(child)

        replace_linear(self.model)
    
    def load_from_path(self, path):
        logger.info(f"Loading quantized model from {path}")
        try:
            state_dict = torch.load(path, map_location=self.device)
            self._quantized_state_dict = state_dict
            self._quantized = True
            
            # Replace all linear layers with WeightOnlyInt4Linear
            self.replace_all_linear_layers()
            
            # Load the state dict
            self.model.load_state_dict(state_dict, strict=False)
            
            logger.info("Successfully loaded quantized model")
            return self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            raise

    def make_names_and_values_dict_func(self, q, qparams):
        packed_weights = pack_int4_weights(q).t()
        scales, zeros = qparams
        #make sure that scales and zeros are a good shape, but it looks good otherwise for the weights.
        return {"weight": packed_weights, "scales": scales, "zeros": zeros}

    def quantize(self, batch_size: int = 1):
        if self.quantized_state_dict_path and os.path.exists(self.quantized_state_dict_path):
            return self.load_from_path(self.quantized_state_dict_path)

        logger.info("Starting GPTQ quantization process...")
        self.model.eval()
        layers = self.find_block_module_list()
        
        examples = self.calibration_data_fn()
        layer_inputs = []

        # Initial forward pass to get inputs for the first layer
        def first_layer_hook(module, input, output):
            layer_inputs.append(input[0].detach())

        first_layer = layers[0]
        first_layer_hook_handle = first_layer.register_forward_hook(first_layer_hook)

        with torch.no_grad():
            for example in examples:
                self.model(example.to(self.device))

        first_layer_hook_handle.remove()

        for i, layer in enumerate(layers):
            logger.info(f"Quantizing layer {i + 1}/{len(layers)} : {get_current_time_string()}")
            
            layer_state_dict = {}
            modules_to_quantize = {}
            for module_paths in self.quantize_config["inside_layer_modules"]:
                for module_path in module_paths:
                    module = get_nested_attr(layer, module_path)
                    if module is not None:
                        modules_to_quantize[module_path] = module
            
            quantizers = {}
            for name, module in modules_to_quantize.items():
                quantizers[name] = GPTQLinearModuleQuantizer(
                    module, 
                    blocksize=self.quantize_config.get("blocksize", 128),
                    percdamp=self.quantize_config.get("percdamp", 0.01),
                    groupsize=self.quantize_config["groupsize"],
                    device=self.device
                )
            
            # Attach hooks to all modules for quantization
            quant_hooks = []
            for name, module in modules_to_quantize.items():
                def quant_hook_fn(module, input, output, name=name):
                    quantizers[name].add_batch(input[0], output)
                
                quant_hooks.append(module.register_forward_hook(quant_hook_fn))
            
            # Pass layer_inputs through the current layer
            with torch.no_grad():
                for inp in layer_inputs:
                    layer(inp)
            # Remove hooks
            for hook in quant_hooks:
                hook.remove()
            
            # Quantize and update state dict
            for name, quantizer in quantizers.items():
                logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)} : {get_current_time_string()}")
                Q, _, QParams = quantizer.quantize()
                
                names_and_values_dict = self.make_names_and_values_dict_func(Q, QParams)
                self.update_quantized_state_dict(name, i, names_and_values_dict)
                layer_state_dict[name] = names_and_values_dict
                
                # Free memory
                quantizer.free()
                del quantizer
            
            # Clear unnecessary memory
            del quantizers
            
            # Replace the original modules with quantized versions
            self.replace_modules_in_layer(layer, modules_to_quantize, layer_state_dict)
            
            # Pass inputs through the quantized layer to get new inputs for the next layer
            layer_outputs = []
            with torch.no_grad():
                for inp in layer_inputs:
                    output = layer(inp)
                    layer_outputs.append(output[0].detach())
            
            layer_inputs = layer_outputs
            
            # Clear layer_state_dict
            del layer_state_dict
            
            torch.cuda.empty_cache()
        
        self._quantized = True
        
        if self.quantized_state_dict_path:
            torch.save(self._quantized_state_dict, self.quantized_state_dict_path)
        
        logger.info("GPTQ quantization process completed.")
        return self.model.to(self.device)

    def update_quantized_state_dict(self, name:str, index:int, module_state_dict:dict) -> None:
        prefix = ".".join(self.quantize_config["path_to_blocks"])
        for key, value in module_state_dict.items():
            self._quantized_state_dict[prefix + "." + str(index) + "." + name + "." + key] = value

    def find_block_module_list(self) -> nn.ModuleList:
        assert "path_to_blocks" in self.quantize_config, "You must specify how the blocks are accessed in your transformer."
        module = self.model
        path_to_blocks = self.quantize_config["path_to_blocks"]
        for child_module in path_to_blocks:
            module = getattr(module, child_module)
        return module

    def replace_modules_in_layer(self, layer, modules_to_quantize, layer_state_dict):
        for name, module in modules_to_quantize.items():
            if isinstance(module, (nn.Linear, transformers.pytorch_utils.Conv1D)):
                if isinstance(module, transformers.pytorch_utils.Conv1D):
                    in_features, out_features = module.weight.shape[0], module.weight.shape[-1]
                else:  # nn.Linear
                    in_features, out_features = module.weight.shape[-1], module.weight.shape[0]
                
                groupsize = self.quantize_config["groupsize"]
                inner_k_tiles = groupsize // 16
                
                if check_linear_int4_k(in_features, groupsize, inner_k_tiles) or self.quantize_config.get("padding_allowed", True):
                    new_module = WeightOnlyInt4Linear(
                        in_features, out_features, bias=False,
                        groupsize=groupsize, inner_k_tiles=inner_k_tiles
                    )
                    new_module.load_state_dict(layer_state_dict[name])
                    
                    # Set the new module in the layer
                    parts = name.split('.')
                    current = layer
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], new_module.to(self.device))