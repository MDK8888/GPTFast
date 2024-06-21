import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model
import torch
from GPTFast.Core import add_speculative_decoding
from GPTFast.Core import KVCacheModel
from GPTFast.Helpers import timed

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.view(1, 1)

# Example usage
model_name = "gpt2-xl"
draft_model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

config = {
    "model_config": {
        "path_to_blocks": ["transformer", "h"],
        "child_ref_in_parent_forward": ["transformer", "block"],
    },
    "block_config": {
        "path_to_attn": ["attn"],
        "child_ref_in_parent_forward": ["attn"], 
    },
    "attn_config": {
        "cache_update_config":{
            "kv_cache_condition":"if layer_past is not None",
            "key_name": "key",
            "value_name": "value",
        },
        "causal_mask_config": {
            "causal_mask_application": "conditional",
            "causal_mask_method": "_attn",
            "causal_mask_condition": "not self.is_cross_attention"
        }
    },
    "imports": ["import torch", 
                "import transformers", 
                "from transformers import *", 
                "from torch import *", 
                "from typing import *", 
                "import types", 
                "from transformers.modeling_outputs import *", 
                "from torch import nn"] 
}

model = KVCacheModel.add_static_cache_to_model(model, cache_config=config, max_generated_length=60, dtype=torch.float32, device="cpu")
draft_model = KVCacheModel.add_static_cache_to_model(draft_model, cache_config=config, max_generated_length=60, dtype=torch.float32, device="cpu")

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

model = add_speculative_decoding(model, draft_model)

N_ITERS=10
MAX_TOKENS=50

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"speculative decode eval time {i}: {compile_time}")
print("~" * 10)
