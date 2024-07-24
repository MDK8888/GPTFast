import os
import copy
import types
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
from GPTFast.Helpers.Time import *

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

        self.config = self.model.config
        self.calibration_data_fn = types.MethodType(calibration_data_fn, self)
        self.quantized_state_dict = self.model.state_dict()
        self.quantize_config = quantize_config
        model_suffix = self.model_name.split("/")[-1]
        self.quantized_state_dict_path = f"{model_suffix}-gptq.pth" if self.quantize_config["save_quantized_state_dict"] else None
        self.device = device
        self.model = self.model.to(self.device)
        self._quantized = False
    
    def skip_layer_func(self, name:str, linear_module:Union[nn.Linear, transformers.pytorch_utils.Conv1D]) -> bool:
        linear_weight = linear_module.weight
        return (name in self.quantize_config["skipped_layers"]) or (not check_linear_int4_k(linear_weight.shape[-1], self.quantize_config["groupsize"]))

    def _prepare_examples_for_quantization(self, examples:List[Dict[str, Union[List[int], torch.LongTensor]]], batch_size:int):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_examples = []
        for example in examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id

        new_examples = [
            collate_data(new_examples[start : start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples
    
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

    @torch.inference_mode()
    def create_quantized_state_dict(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
    ):

        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

        examples = self._prepare_examples_for_quantization(examples, batch_size)

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = self.find_block_module_list() 

        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(inp.to(self.device))
            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs["attention_mask"] is not None:
                attention_masks.append(kwargs["attention_mask"].to(self.device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(pos_ids.to(self.device))
            one_kwargs = {}
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for example in examples:
            try:
                for k, v in example.items():
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = v.to(self.device)
                #the forward hook runs before each forward pass. Basically, after this loop runs, we obtain args, kwargs, attention_masks, and position_ids.
                #we also move everything to the cur_layer_device. 
                self.model(**example)
            except ValueError:
                pass
        handle.remove()

        torch.cuda.empty_cache()

        #inside_layer_modules is just the modules inside of each block.
        inside_layer_modules = self.quantize_config["inside_layer_modules"]
        if not self.quantize_config["true_sequential"]:
            inside_layer_modules = [sum(inside_layer_modules, [])] #for GPT2 for example, this is c_attn, c_proj, mlp.c_fc, mlp.c_proj
        for i in range(len(layers)):
            logger.info(f"Start quantizing layer {i + 1}/{len(layers)} {get_current_time_string()}")
            layer = layers[i]

            full = find_layers_dict(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names if n in full}
                #so, subset is just the modules we want to quantize in a dict, where the names are strings and the values are the layer itself.
                #n is the name of the string.
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQLinearModuleQuantizer(subset[name], groupsize=self.quantize_config["groupsize"], device=self.device)

                #this is a little sus-we return a function after the forward method? 
                #Got you - basically, when add_batch returns a function, the function returned is the actual hook that will run. 
                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handles = []
                #the whole point of these next 3 for-loops is to add_batches to all of our GPTQ quantizers so that we can quantize later. 
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = []
                    for k, layer_inp in enumerate(layer_inputs[j]):
                        layer_input.append(layer_inp.to(self.device))

                    layer_attention_mask = attention_masks[j].to(self.device)
                    additional_layer_inputs = {"attention_mask": layer_attention_mask}
                    layer(*layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                #actually quantize each layer.
                for name in subset:
                    logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)} {get_current_time_string()}")
                    Q, DQ, QParams = gptq[name].quantize()

                    #modify state_dict here.
                    names_and_values_dict = self.make_names_and_values_dict_func(Q, QParams)
                    self.update_quantized_state_dict(name, i, names_and_values_dict)
                    gptq[name].free()

            #finally, we get the layer_input for the next block by passing in the layer_output from the previous block.
            for j in range(num_batches):
                layer_input = []
                for k, layer_inp in enumerate(layer_inputs[j]):
                    layer_input.append(layer_inp.to(self.device))

                layer_attention_mask = attention_masks[j].to(self.device)
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_output = layer(*layer_input, **additional_layer_inputs)[0].to(self.device)
                layer_outputs.append([layer_output])

            layers[i] = layer.to(self.device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []  # TODO: is it really OK to cache only the first positional argument?
            torch.cuda.empty_cache()

        self.model.config.use_cache = forward_pass_use_cache
        if self.quantized_state_dict_path != None:
            torch.save(self.quantized_state_dict, self.quantized_state_dict_path)

        self._quantized = True

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
                        groupsize=groupsize, inner_k_tiles=inner_k_tiles,
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
        else:
            logger.info("The quantized_state_dict_path is None - quantized_state_dict will not be saved...")
            examples = self.calibration_data_fn()
            self.create_quantized_state_dict(examples, batch_size)

        logger.info("Swapping nn.Linear layers out and replacing them with WeightOnlyInt4Linear...")
        groupsize = self.quantize_config["groupsize"]
        self.replace_linear_int4(self.model, groupsize=groupsize, inner_k_tiles=groupsize // 16, skip_layer_func=self.skip_layer_func)
        logger.info("Loading quantized_state_dict into the model...")
        self.model.load_state_dict(self.quantized_state_dict)
        self.model = self.model.to(self.device)
        return self.model