import torch
from transformers import AutoModelForCausalLM
from typing import Union, Dict
from .GPTQ import *
from .INT8 import *

def quantize(quantization_mode:str, model_name:str, calibration_data_fn:Callable[..., Dict[str, Union[torch.LongTensor, list[int]]]], quantize_config:dict, device:torch.device):
    if quantization_mode == None:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model
    elif quantization_mode == "INT8":
        quantizer = Int8Quantizer(model_name=model_name, device=device)
    elif quantization_mode == "GPTQ":
        quantizer = GPTQModelQuantizer(model_name=model_name, calibration_data_fn=calibration_data_fn, quantize_config=quantize_config, device=device)
    else:
        raise ValueError("This quantization format is currently unsupported.")
    model = quantizer.quantize()
    return model    