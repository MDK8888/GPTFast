import torch
from torch import nn
from transformers import AutoModelForCausalLM
from ...Quantizer import Quantizer

class Int8Quantizer(Quantizer):
    
    def __init__(self, model_name:str, device:torch.device):
        self.model_name = model_name
        self.device = device

    def quantize(self) -> nn.Module:
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True)
        return self.model
