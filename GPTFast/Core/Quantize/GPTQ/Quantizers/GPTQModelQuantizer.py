import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from typing import Dict, Union, Callable
import logging
from GPTFast.Helpers import get_nested_attr, set_nested_attr
from .GPTQLinearModuleQuantizer import GPTQLinearModuleQuantizer
from ..Modules import WeightOnlyInt4Linear
from GPTFast.Kernels import pack_int4_weights

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GPTQModelQuantizer:
    def __init__(self, model_name: str, calibration_data_fn: Callable, quantize_config: dict, device: torch.device = "cuda"):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
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

    def quantize_layer(self, layer, name, stored_data):
        print(f"Quantizing layer: {name}")
        quantizer = GPTQLinearModuleQuantizer(layer, name, blocksize=self.quantize_config.get("blocksize", 128),
                                              percdamp=self.quantize_config.get("percdamp", 0.01),
                                              groupsize=self.quantize_config["groupsize"],
                                              device=self.device, dtype=torch.float32)

        for input_data, output_data in stored_data[name]:
            quantizer.add_batch(input_data, output_data)
        
        Q, DQ, qparams = quantizer.quantize()
        
        quantizer.free()
        del quantizer

        scales, zeros = qparams
        
        packed_weights = pack_int4_weights(Q).t().contiguous()

        return packed_weights, scales, zeros, layer.bias.clone().detach() if self.quantize_config["has_bias"] else None

    def process_single_layer(self, name, layer, stored_data):
        print(f"\nProcessing layer: {name}", flush=True)
        
        packed_weights, scales, zeros, bias = self.quantize_layer(layer, name, stored_data)
        if packed_weights is None:
            logger.warning(f"Quantization failed for {name}")
            return None

        errors = {'mse': [], 'relative_error': []}

        in_features, out_features = layer.weight.shape[1], layer.weight.shape[0]
        weight_only_int4_linear = WeightOnlyInt4Linear(in_features, out_features, name, packed_weights, scales, zeros, bias, groupsize=self.quantize_config["groupsize"])
        for input_data, original_output in stored_data[name]:
            with torch.no_grad():
                quantized_output_kernel = weight_only_int4_linear(input_data)

            mse = F.mse_loss(original_output, quantized_output_kernel).item()
            relative_error = torch.norm(original_output - quantized_output_kernel) / torch.norm(original_output)
            errors['mse'].append(mse)
            errors['relative_error'].append(relative_error.item())

        if self.quantize_config.get("log_quant_stats", False):
            print(f"Quantization results for {name}:", flush=True)
            avg_mse = sum(errors['mse']) / len(errors['mse'])
            avg_relative_error = sum(errors['relative_error']) / len(errors['relative_error'])
            print(f"  Average MSE: {avg_mse}", flush=True)
            print(f"  Average Relative Error: {avg_relative_error}", flush=True)

        print(f"Finished processing layer: {name}", flush=True)
        return packed_weights, scales, zeros, bias, errors

    def quantize(self):
        print("Starting quantization process...")
        
        if os.path.exists(self.quantized_state_dict_path):
            print(f"Loading existing quantized weights from {self.quantized_state_dict_path}", flush=True)
            return self.load_quantized_model()
        
        print("No existing quantized weights found. Performing quantization...")
        
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
            print(f"Quantizing block {i + 1}/{len(blocks)}", flush=True)
            
            layers_to_quantize = {}
            for module_paths in self.quantize_config["inside_layer_modules"]:
                    for module_path in module_paths:
                        module = get_nested_attr(block, module_path)
                        if module is not None and isinstance(module, (nn.Linear, nn.Conv1d)) and not any(skip in module_path for skip in self.quantize_config.get("skip_layers", [])):
                            layers_to_quantize[module_path] = module

            stored_data = {name: [] for name in layers_to_quantize}

            def hook_fn(name):
                def hook(module, input, output):
                    stored_data[name].append((input[0].detach(), output.detach()))
                return hook

            hooks = [module.register_forward_hook(hook_fn(name)) for name, module in layers_to_quantize.items()]

            # Pass block inputs through the block
            block_outputs = []
            with torch.no_grad():
                for block_input in block_inputs:
                    block_output = block(block_input)
                    block_outputs.append(block_output[0])
            
            block_inputs = block_outputs

            for hook in hooks:
                hook.remove()

            stored_data_copy = copy.deepcopy(stored_data)

            print("\nQuantizing layers and saving weights...")
            for name, layer in layers_to_quantize.items():
                packed_weights, scales, zeros, bias, _ = self.process_single_layer(name, layer, stored_data_copy)

                if packed_weights is not None:
                    full_name = f"{'.'.join(self.quantize_config['path_to_blocks'])}.{i}.{name}"
                    self.quantized_state_dict[full_name] = {
                        'weight': packed_weights.cpu(),
                        'scales': scales.cpu(),
                        'zeros': zeros.cpu(),
                        'bias': bias.cpu() if bias is not None else None
                    }

            del stored_data
            del stored_data_copy

        self._quantized = True
        
        # Save the quantized state dict
        self.save_quantized_model()
        return self.load_quantized_model()

    def replace_linear_layers(self):
        blocks = self.find_block_module_list()
        for i, block in enumerate(blocks):
            for name, module in block.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)) and not any(skip in name for skip in self.quantize_config.get("skip_layers", [])):
                    full_name = f"{'.'.join(self.quantize_config['path_to_blocks'])}.{i}.{name}"
                    if full_name in self.quantized_state_dict:
                        state_dict = self.quantized_state_dict[full_name]
                        in_features, out_features = module.weight.shape[1], module.weight.shape[0]
                        if isinstance(module, nn.Conv1d):
                            in_features, out_features = out_features, in_features
                        
                        new_module = WeightOnlyInt4Linear(
                            in_features=in_features,
                            out_features=out_features,
                            name=name,
                            weight=state_dict['weight'].to(self.device),
                            scales=state_dict['scales'].to(self.device),
                            zeros=state_dict['zeros'].to(self.device),
                            bias=state_dict['bias'].to(self.device) if state_dict['bias'] is not None else None,
                            groupsize=self.quantize_config["groupsize"]
                        )

                        set_nested_attr(block, name, new_module)

    def save_quantized_model(self):
        if not self._quantized:
            logger.warning("Model has not been quantized yet. Call quantize() first.")
            return
        
        torch.save({
            'model_state_dict': self.quantized_state_dict,
            'config': self.quantize_config
        }, self.quantized_state_dict_path)
        print(f"Quantized model saved to {self.quantized_state_dict_path}")

    def load_quantized_model(self):
        checkpoint = torch.load(self.quantized_state_dict_path)
        self.quantized_state_dict = checkpoint['model_state_dict']
        self.quantize_config = checkpoint['config']
        self._quantized = True
        
        self.replace_linear_layers()
        print(f"Quantized model loaded from {self.quantized_state_dict_path}")
        return self.model