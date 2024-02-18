from transformers import AutoModelForCausalLM
from .KVCache import add_kv_cache
from .Compile import torch_compile_model
from .Quantize import *
from .SpeculativeDecode import add_speculative_decoding
from .Quantize import load_int8

def gpt_fast(model_name:str, **spec_dec_kwargs):
    model = load_int8(model_name)
    model = add_kv_cache(model)
    spec_decode = False
    if spec_dec_kwargs:
        draft_model_name = spec_dec_kwargs.pop("draft_model_name")
        draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
        draft_model = add_kv_cache(draft_model)
        model = add_speculative_decoding(model, draft_model, **spec_dec_kwargs)
        spec_decode = True
    model = torch_compile_model(model, spec_decode=spec_decode)
    return model
