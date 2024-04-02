from typing import Callable
import torch
from transformers import AutoModelForCausalLM
from GPTFast.Core.KVCache import add_kv_cache
from GPTFast.Core.Compile import torch_compile_model
from GPTFast.Core.Decode import add_speculative_decoding
from GPTFast.Core.Quantize import load_int8

def gpt_fast(model_name:str, sample_function:Callable, max_length:int, cache_config:dict, **spec_dec_kwargs):
    model = load_int8(model_name)
    model = add_kv_cache(model, sample_function, max_length, cache_config, dtype=torch.float16)
    spec_decode = False
    if spec_dec_kwargs:
        draft_model_name = spec_dec_kwargs.pop("draft_model_name")
        draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
        draft_model = add_kv_cache(draft_model, sample_function, max_length, cache_config, dtype=torch.float32)
        model = add_speculative_decoding(model, draft_model, **spec_dec_kwargs)
        spec_decode = True
    model = torch_compile_model(model, spec_decode=spec_decode)
    return model