from torch import nn
from transformers import AutoModelForCausalLM

def load_int8(model_name:str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model