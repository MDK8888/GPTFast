import os
import copy
import json
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

logger = getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#note: For GPT2, position_ids is not important since it just uses positional embeddings, but for Llama2 it is important due to the RoPE Embedding. 
#We will deal with that later.

class GPTQModelQuantizer(Quantizer):

    def __init__(self, model_name:str, calibration_data_fn:Callable[..., Dict[str, Union[torch.LongTensor, list[int]]]], quantize_config:dict, device:torch.device = "cpu"):
        self.model_name = model_name
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        except Exception as e:
            logger.info("We do not need to specify the attention implementation for this model, using default...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model = self.model.to(torch.bfloat16)
        self.config = self.model.config
        self.calibration_data_fn = types.MethodType(calibration_data_fn, self)
        self.quantized_state_dict = self.model.state_dict()
        self.quantize_config = quantize_config
        model_suffix = self.model_name.split("/")[-1]
        self.quantized_state_dict_path = f"{model_suffix}-gptq.pth" if self.quantize_config["save_quantized_state_dict"] else None
        self.device = device
        self.model = self.model.to(self.device)
        self.json_log_path = f"{model_suffix}-gptq-logs.json"
        self.log_quant_stats = quantize_config.get("log_quant_stats", False)
        self.quant_stats = defaultdict(lambda: defaultdict(list))
        self._quantized = False
    
    def skip_layer_func(self, name:str, linear_module:Union[nn.Linear, transformers.pytorch_utils.Conv1D]) -> bool:
        linear_weight = linear_module.weight
        return (name in self.quantize_config["skipped_layers"]) or (not check_linear_int4_k(linear_weight.shape[-1], self.quantize_config["groupsize"]))
    
    def make_names_and_values_dict_func(self, q, qparams):
        groupsize = self.quantize_config["groupsize"]
        k = q.shape[1]
        if not check_linear_int4_k(k, groupsize):
            new_k = find_multiple(k, 1024)
        else:
            new_k = k
        # how much we need to pad the weight
        delta_k = new_k - q.shape[1]
        q = q.to(torch.int32).to(self.device)
        final_q = torch.ops.aten._convert_weight_to_int4pack(F.pad(q, pad=(0, delta_k)).contiguous(), innerKTiles = groupsize // 16)
        scales = qparams[0].to(torch.bfloat16).to(self.device)
        zeros = qparams[1].to(torch.bfloat16).to(self.device)
        scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)

        # how many new groups we need for padded weight
        delta_groups = new_k // self.quantize_config["groupsize"] - scales_and_zeros.shape[0]
        final_s_and_z = F.pad(scales_and_zeros, pad=(0,0,0,0,0, delta_groups), value=1)
        return {"weight": final_q, "scales_and_zeros": final_s_and_z}
    
    def update_quantized_state_dict(self, name:str, index:int, module_state_dict:dict) -> None:
        prefix = ".".join(self.quantize_config["path_to_blocks"])
        if name + ".weight" in self.quantized_state_dict:
            self.quantized_state_dict.pop(name + ".weight")
        if prefix + "." + str(index) + "." + name + ".bias" in self.quantized_state_dict:
            self.quantized_state_dict.pop(prefix + "." + str(index) + "." + name + ".bias")
        for key, value in module_state_dict.items():
            self.quantized_state_dict[prefix + "." + str(index) + "." + name + "." + key] = value

    def find_block_module_list(self) -> nn.ModuleList:
        assert "path_to_blocks" in self.quantize_config, "You must specify how the blocks are accessed in your transformer."
        module = self.model
        path_to_blocks = self.quantize_config["path_to_blocks"]
        for child_module in path_to_blocks:
            module = getattr(module, child_module)

        return module

    @torch.inference_mode
    def create_quantized_state_dict(self, examples: list[torch.Tensor], batch_size: int = 1):
        self.model.eval()
        layers = self.find_block_module_list()
        
        layer_input = []
        layer_output = []

        # Hook for the first layer to collect inputs
        def first_layer_hook(module, input, output):
            layer_input.append(input[0].detach())

        first_layer = layers[0]
        first_layer_hook_handle = first_layer.register_forward_hook(first_layer_hook)

        # Pass examples through the first layer to populate layer_input
        with torch.no_grad():
            for example in examples:
                self.model(example)

        first_layer_hook_handle.remove()

        for i, layer in enumerate(layers):
            logger.info(f"Quantizing layer {i + 1}/{len(layers)} : {get_current_time_string()}")
            
            # Find the modules to quantize within this layer
            modules_to_quantize = {}
            for module_paths in self.quantize_config["inside_layer_modules"]:
                for module_path in module_paths:
                    module = get_nested_attr(layer, module_path)
                    if module is not None:
                        modules_to_quantize[module_path] = module
            
            # Create GPTQLinearModuleQuantizers for each module
            quantizers = {}
            if self.log_quant_stats:
                original_outputs = {}
            for name, module in modules_to_quantize.items():
                quantizers[name] = GPTQLinearModuleQuantizer(
                    module, 
                    blocksize=self.quantize_config.get("blocksize", 128),
                    percdamp=self.quantize_config.get("percdamp", 0.01),
                    groupsize=self.quantize_config["groupsize"],
                    device=self.device
                )
                if self.log_quant_stats:
                    original_outputs[name] = []
            
            # Attach hooks to all modules for quantization and collecting original outputs
            quant_hooks = []
            for name, module in modules_to_quantize.items():
                def quant_hook_fn(module, input, output, name=name):
                    quantizers[name].add_batch(input[0], output)
                    if self.log_quant_stats:
                        original_outputs[name].append(output.detach())
                
                quant_hooks.append(module.register_forward_hook(quant_hook_fn))
            
            # Hook for the current layer to collect outputs
            def layer_output_hook(module, input, output):
                layer_output.append(output[0].detach())

            layer_output_hook_handle = layer.register_forward_hook(layer_output_hook)
            
            # Pass layer_input through the current layer
            with torch.no_grad():
                for inp in layer_input:
                    layer(inp)
            
            # Remove hooks
            for hook in quant_hooks:
                hook.remove()
            layer_output_hook_handle.remove()
            
            # Quantize and update state dict
            for name, quantizer in quantizers.items():
                logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)} : {get_current_time_string()}")
                Q, DQ, QParams = quantizer.quantize()
                
                names_and_values_dict = self.make_names_and_values_dict_func(Q, QParams)
                self.update_quantized_state_dict(name, i, names_and_values_dict)
                
                if self.log_quant_stats:
                    # Calculate activation error
                    quantized_module = WeightOnlyInt4Linear(
                        quantizer.layer.weight.shape[0], 
                        quantizer.layer.weight.shape[1], 
                        bias=False,
                        groupsize=self.quantize_config["groupsize"],
                        inner_k_tiles=self.quantize_config["groupsize"] // 16
                    ).to(self.device)
                    quantized_module.load_state_dict(names_and_values_dict)
                    
                    total_mse = 0
                    total_relative_error = 0
                    num_samples = len(quantizer.all_inputs)

                    for original_input, original_output in zip(quantizer.all_inputs, original_outputs[name]):
                        # Ensure inputs have the same shape
                        if original_input.dim() == 2:
                            original_input = original_input.unsqueeze(0)  # Add batch dimension if needed
                        
                        quantized_output = quantized_module(original_input)
                        
                        # Ensure outputs have the same shape for comparison
                        if original_output.dim() == 3 and quantized_output.dim() == 2:
                            quantized_output = quantized_output.unsqueeze(0)
                        elif original_output.dim() == 2 and quantized_output.dim() == 3:
                            original_output = original_output.unsqueeze(0)
                        
                        # Calculate MSE
                        mse = F.mse_loss(original_output, quantized_output).item()
                        
                        # Calculate relative error
                        relative_error = torch.norm(original_output - quantized_output) / torch.norm(original_output)
                        
                        total_mse += mse
                        total_relative_error += relative_error.item()

                    avg_mse = total_mse / num_samples
                    avg_relative_error = total_relative_error / num_samples

                    logger.info(f"  Activation error for {name}:")
                    logger.info(f"    Average MSE: {avg_mse:.6f}")
                    logger.info(f"    Average Relative Error: {avg_relative_error:.6f}")
                
                # Free memory
                quantizer.free()
                del quantizer
            
            # Clear unnecessary memory
            del quantizers
            del layer_input
            layer_input = layer_output
            layer_output = []
            torch.cuda.empty_cache()
        
        self._quantized = True
        
        if self.quantized_state_dict_path:
            torch.save(self.quantized_state_dict, self.quantized_state_dict_path)
        
        torch.cuda.empty_cache()

    def replace_linear_int4(self, module:nn.Module, groupsize:int = 128, inner_k_tiles:int = 8, padding_allowed:bool = True, skip_layer_func:Callable = None):
        #logger.info("Swapping nn.Linear layers out and replacing them with WeightOnlyInt4Linear")
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, transformers.pytorch_utils.Conv1D)) and (skip_layer_func is None or not skip_layer_func(name, child)):
                if isinstance(child, transformers.pytorch_utils.Conv1D): in_features, out_features = child.weight.shape[0], child.weight.shape[-1]
                else: in_features, out_features = child.weight.shape[-1], child.weight.shape[0]
                if check_linear_int4_k(in_features, groupsize, inner_k_tiles) or padding_allowed:
                    #logger.info(f"Transforming {name} from {child} to WeightOnlyInt4Linear")
                    setattr(module, name, WeightOnlyInt4Linear(
                        in_features, out_features, bias=False,
                        groupsize=groupsize, inner_k_tiles=inner_k_tiles
                    ))
            else:
                self.replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, skip_layer_func)
    
    def quantize(self, batch_size:int = 1):
        if self.quantized_state_dict_path != None:
            if not os.path.exists(self.quantized_state_dict_path):
                logger.info(f"The specified quantized_state_dict_path '{self.quantized_state_dict_path}' does not currently exist. Creating quantized_state_dict and saving it at: {self.quantized_state_dict_path}...")
                examples = self.calibration_data_fn()
                self.create_quantized_state_dict(examples, batch_size)
            else:
                logger.info(f"Quantized_state_dict_path already exists - loading the quantized_state_dict at: {self.quantized_state_dict_path}...")
                self.quantized_state_dict = torch.load(self.quantized_state_dict_path)
                self._quantized = True
        else:
            logger.info("The quantized_state_dict_path is None - quantized_state_dict will not be saved...")
            examples = self.calibration_data_fn()
            self.create_quantized_state_dict(examples, batch_size)

        logger.info("Swapping nn.Linear layers out and replacing them with WeightOnlyInt4Linear...")
        groupsize = self.quantize_config["groupsize"]
        self.replace_linear_int4(self.model, groupsize=groupsize, inner_k_tiles=groupsize // 16, skip_layer_func=self.skip_layer_func)
        logger.info("Loading quantized_state_dict into the model...")
        self.model.load_state_dict(self.quantized_state_dict)

        return self.model.to(self.device)