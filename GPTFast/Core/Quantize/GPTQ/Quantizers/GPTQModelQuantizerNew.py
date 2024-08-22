import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from tqdm.auto import tqdm
from typing import Dict, Union, Callable
import logging
from GPTFast.Helpers import get_nested_attr
from .GPTQLinearModuleQuantizer import GPTQLinearModuleQuantizer
from ..Modules import WeightOnlyInt4Linear
from GPTFast.Kernels import pack_int4_weights

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GPTQModelQuantizerNew:
    def __init__(self, model_name: str, calibration_data_fn: Callable, quantize_config: dict, device: torch.device = "cuda"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.calibration_data_fn = calibration_data_fn
        self.quantize_config = quantize_config
        self.device = device
        self.quantized_state_dict = {}
        self._quantized = False
        model_suffix = self.model_name.split("/")[-1]
        self.quantized_state_dict_path = f"{model_suffix}-gptq.pth"

    def find_block_module_list(self):
        assert "path_to_blocks" in self.quantize_config, "You must specify how the blocks are accessed in your transformer."
        module = self.model
        path_to_blocks = self.quantize_config["path_to_blocks"]
        for child_module in path_to_blocks:
            module = getattr(module, child_module)
        return module

    def calculate_error(self, original_output, quantized_output):
        mse = F.mse_loss(original_output, quantized_output).item()
        relative_error = torch.norm(original_output - quantized_output) / torch.norm(original_output)
        return mse, relative_error.item()

    def quantize_layer(self, layer, name, stored_data):
        logger.info(f"Quantizing layer: {name}")
        quantizer = GPTQLinearModuleQuantizer(layer, name, blocksize=self.quantize_config.get("blocksize", 128),
                                              percdamp=self.quantize_config.get("percdamp", 0.01),
                                              groupsize=self.quantize_config["groupsize"],
                                              device=self.device, dtype=torch.float16)

        for input_data, output_data in stored_data[name]:
            quantizer.add_batch(input_data, output_data)
        
        Q, DQ, qparams = quantizer.quantize()

        quantizer.free()
        del quantizer
        
        scales, zeros = qparams
        packed_weights = pack_int4_weights(Q).t().contiguous()
        
        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        if isinstance(layer, Conv1D):
            in_features, out_features = out_features, in_features
        
        weight_only_int4_linear = WeightOnlyInt4Linear(in_features, out_features, groupsize=self.quantize_config["groupsize"]).to(self.device)
        weight_only_int4_linear.weight = packed_weights
        weight_only_int4_linear.scales = scales.to(torch.float16)
        weight_only_int4_linear.zeros = zeros.to(torch.float16)
        weight_only_int4_linear.bias = layer.bias.clone() if layer.bias is not None else None
        
        quantizer.free()

        return weight_only_int4_linear, DQ

    def process_single_layer(self, name, layer, stored_data):
        logger.info(f"\nProcessing layer: {name}")
        
        weight_only_int4_linear, DQ = self.quantize_layer(layer, name, stored_data)
        
        errors = {'mse': [], 'relative_error': []}
        
        for input_data, original_output in stored_data[name]:
            input_data = input_data.to(torch.float16)
            original_output = original_output.to(torch.float16)
            with torch.no_grad():
                quantized_output_kernel = weight_only_int4_linear(input_data)
            
                mse, relative_error = self.calculate_error(original_output, quantized_output_kernel)
                errors['mse'].append(mse)
                errors['relative_error'].append(relative_error)
        
        logger.info(f"Finished processing layer: {name}")
        return weight_only_int4_linear, DQ, errors

    def quantize(self, batch_size: int = 1):
        logger.info("Starting quantization process...")
        
        if os.path.exists(self.quantized_state_dict_path):
            logger.info(f"Loading existing quantized weights from {self.quantized_state_dict_path}")
            return self.load_quantized_model()
        
        logger.info("No existing quantized weights found. Performing quantization...")
        
        examples = self.calibration_data_fn(self.tokenizer)
        blocks = self.find_block_module_list()
        
        # Collect inputs to the first block
        first_block_inputs = []
        def first_block_hook(module, input, output):
            first_block_inputs.append(input[0].detach())
        
        first_block_hook_handle = blocks[0].register_forward_hook(first_block_hook)
        
        # Pass examples through the model to collect first block inputs
        self.model.eval()
        with torch.no_grad():
            for example in tqdm(examples, desc="Collecting first block inputs"):
                self.model(example.to(self.device))
        
        first_block_hook_handle.remove()
        
        # Quantize blocks
        block_inputs = first_block_inputs
        for i, block in enumerate(tqdm(blocks, desc="Quantizing blocks")):
            logger.info(f"Quantizing block {i + 1}/{len(blocks)}")
            
            modules_to_quantize = {}
            stored_data = {}
            
            for module_paths in self.quantize_config["inside_layer_modules"]:
                for module_path in module_paths:
                    module = get_nested_attr(block, module_path)
                    if module is not None and isinstance(module, (nn.Linear, Conv1D)) and not any(skip in module_path for skip in self.quantize_config.get("skip_layers", [])):
                        modules_to_quantize[module_path] = module
                        stored_data[module_path] = []

            def hook_fn(name):
                def hook(module, input, output):
                    stored_data[name].append((input[0].detach(), output.detach()))
                return hook

            hooks = [module.register_forward_hook(hook_fn(name)) for name, module in modules_to_quantize.items()]

            # Pass block inputs through the block
            with torch.no_grad():
                for input_tensor in block_inputs:
                    block(input_tensor)

            for hook in hooks:
                hook.remove()

            stored_data_copy = copy.deepcopy(stored_data)
            # Quantize and replace modules
            for name, module in modules_to_quantize.items():
                weight_only_int4_linear, _, errors = self.process_single_layer(name, module, stored_data_copy)
                
                if weight_only_int4_linear is not None:
                    # Replace the module in the block
                    parent_name, child_name = name.rsplit('.', 1)
                    parent_module = get_nested_attr(block, parent_name)
                    setattr(parent_module, child_name, weight_only_int4_linear)
                    
                    # Update quantized state dict
                    self.quantized_state_dict[f"{'.'.join(self.quantize_config['path_to_blocks'])}.{i}.{name}"] = weight_only_int4_linear.state_dict()
                    
                    # Log quantization statistics
                    avg_mse = sum(errors['mse']) / len(errors['mse'])
                    avg_relative_error = sum(errors['relative_error']) / len(errors['relative_error'])
                    logger.info(f"Quantization statistics for {name}:")
                    logger.info(f"  Average MSE: {avg_mse}")
                    logger.info(f"  Average Relative Error: {avg_relative_error}")
            
            # Prepare inputs for the next block
            block_inputs = [block(input_tensor)[0] for input_tensor in block_inputs]
            del stored_data
            del stored_data_copy
        
        self._quantized = True
        
        # Save the quantized state dict
        self.save_quantized_model()
        return self.model

    def replace_linear_layers(self):
        blocks = self.find_block_module_list()
        for i, block in enumerate(blocks):
            for name, module in block.named_modules():
                if isinstance(module, (nn.Linear, Conv1D)) and not any(skip in name for skip in self.quantize_config.get("skip_layers", [])):
                    full_name = f"{'.'.join(self.quantize_config['path_to_blocks'])}.{i}.{name}"
                    if full_name in self.quantized_state_dict:
                        in_features, out_features = module.weight.shape[1], module.weight.shape[0]
                        if isinstance(module, Conv1D):
                            in_features, out_features = out_features, in_features
                        new_module = WeightOnlyInt4Linear(in_features, out_features, groupsize=self.quantize_config["groupsize"]).to(self.device)
                        new_module.load_state_dict(self.quantized_state_dict[full_name])
                        parent_name, child_name = name.rsplit('.', 1)
                        parent_module = dict(block.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_module)

    def save_quantized_model(self):
        if not self._quantized:
            logger.warning("Model has not been quantized yet. Call quantize() first.")
            return
        
        torch.save({
            'model_state_dict': self.quantized_state_dict,
            'config': self.quantize_config
        }, self.quantized_state_dict_path)
        logger.info(f"Quantized model saved to {self.quantized_state_dict_path}")

    def load_quantized_model(self):
        checkpoint = torch.load(self.quantized_state_dict_path)
        self.quantized_state_dict = checkpoint['model_state_dict']
        self.quantize_config = checkpoint['config']
        self._quantized = True
        
        self.replace_linear_layers()
        logger.info(f"Quantized model loaded from {self.quantized_state_dict_path}")
        return self.model