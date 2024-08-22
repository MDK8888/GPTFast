from typing import Callable, Union, Dict
import torch
from GPTFast.Core.KVCache import add_kv_cache
from GPTFast.Core.Compile import torch_compile_model
from GPTFast.Core.Decode import add_speculative_decoding
from GPTFast.Core.Quantize import quantize

def gpt_fast(model_name:str, calibration_data_function:Callable[..., Dict[str, Union[torch.LongTensor, list[int]]]], quantize_config:dict, \
             sample_function:Callable[..., torch.LongTensor], cache_config:dict, device:torch.device, **spec_dec_kwargs):
    
    #quantize the model
    assert "quantization_mode" in quantize_config, "You must specify how your model will be quantized."
    quantization_mode = quantize_config["quantization_mode"]
    model = quantize(quantization_mode=quantization_mode, model_name=model_name, calibration_data_fn=calibration_data_function, \
                     quantize_config=quantize_config, device=device)
    cache_dtype = torch.float32 if quantization_mode == "GPTQ" else torch.float16

    #Integrate static key-value cache
    model = add_kv_cache(model=model, sample_fn=sample_function, cache_config=cache_config, dtype=cache_dtype, device=device)
    spec_decode = False

    #If we have speculative decoding, apply previous techniques to the draft model as well
    if spec_dec_kwargs:
        draft_model_name = spec_dec_kwargs.pop("draft_model_name")
        draft_quantization_mode = "GPTQ" if quantization_mode == "GPTQ" else None
        draft_cache_dtype = torch.float32
        draft_model = quantize(quantization_mode=draft_quantization_mode, model_name=draft_model_name, calibration_data_fn=calibration_data_function, \
                                   quantize_config=quantize_config, device=device)
        draft_model = add_kv_cache(model=draft_model, sample_fn=sample_function, cache_config=cache_config, dtype=draft_cache_dtype, device=device)

        #add speculative decoding
        model = add_speculative_decoding(model, draft_model, **spec_dec_kwargs)
        spec_decode = True
    
    #torch compile everything at the very end
    model = torch_compile_model(model, spec_decode=spec_decode, device=device)
    return model