import os
import torch
from transformers import AutoTokenizer
from GPTFast.Core import gpt_fast
from GPTFast.Helpers import timed

torch._dynamo.reset()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

def argmax_variation(self, probabilities:torch.Tensor, temperature:float = 1, k:int = 5):
    # Apply temperature scaling
    device = probabilities.device
    scaled_probabilities = probabilities / temperature

    # Ensure k is within a valid range
    k = min(k, probabilities.size(-1))

    # Get the indices of the top-k scaled probabilities along the specified dimension
    top_k_indices = torch.topk(scaled_probabilities, k, dim=-1).indices

    # Generate random indices for sampling
    random_indices = torch.randint(0, k, (1,) * probabilities.dim()).to(device)

    # Use gathered indices to get the final sampled token
    sampled_token = top_k_indices.gather(-1, random_indices).to(device)

    return sampled_token.unsqueeze(0)

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.view(1, 1)

model_name = "gpt2-xl"
draft_model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
initial_string = "Write me a short story."
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

cache_config = {
    "path_to_blocks": ["transformer", "h"],
    "child_ref_in_parent_forward": ["transformer", "block"],
    "attn_ref_in_block_forward": "attn",
    "attention_name": "attn",
    "kv_cache_condition":"layer_past is not None",
    "causal_mask_location": "_attn",
    "causal_mask_condition": "not self.is_cross_attention",
    "key_name": "key",
    "value_name": "value",
    "imports": ["import torch", 
                "import transformers", 
                "from transformers import *", 
                "from torch import *", 
                "from typing import *", 
                "import types", 
                "from transformers.modeling_outputs import *", 
                "from torch import nn"]
}

gpt_fast_model = gpt_fast(model_name, sample_function=argmax, max_length=60, cache_config=cache_config, draft_model_name=draft_model_name)
gpt_fast_model.to(device)

gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6)

'''
fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)

print(tokenizer.decode(res[0]))
'''