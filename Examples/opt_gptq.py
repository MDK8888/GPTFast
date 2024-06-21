import os
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
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

model_name = "facebook/opt-1.3b"
draft_model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
initial_string = "Hello my name is"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

cache_config = {
    "max_length": 60,
    "model_config": {
        "path_to_blocks": ["model", "decoder", "layers"],
        "child_ref_in_parent_forward": ["model.decoder", "decoder", "decoder_layer"],
    },
    "block_config": {
        "path_to_attn": ["self_attn"],
        "child_ref_in_parent_forward": ["self_attn"], 
    },
    "attn_config": {
        "cache_update_config":{
            "kv_cache_condition":"elif past_key_value is not None",
            "key_name": "self._shape(self.k_proj(hidden_states), -1, bsz)",
            "value_name": "self._shape(self.v_proj(hidden_states), -1, bsz)",
            "new_key_name": "key_states",
            "new_value_name": "value_states",
        },
        "causal_mask_config": {
            "causal_mask_application": "conditional",
            "causal_mask_method": "forward",
            "causal_mask_condition": "if attention_mask is not None"
        }
    }, 
    "imports":["import torch", 
               "from torch import nn",
               "from torch import LongTensor",
                "import transformers", 
                "from transformers.pytorch_utils import *",
                "from transformers.modeling_utils import *",
                "from transformers.modeling_outputs import *",
                "from transformers.utils import *",
                "from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask", 
                "from typing import *",  
                "import types", 
            ]
}

def get_wikitext2(self):

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=True)

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    import random

    random.seed(5)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(200):
        i = random.randint(0, trainenc.input_ids.shape[1] - 100 - 1)
        j = i + 100
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset

quantize_config = {
    "quantization_mode": "GPTQ",
    "save_quantized_state_dict": True,
    "groupsize": 32,
    "path_to_blocks": ["model", "decoder", "layers"],
    "inside_layer_modules": [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"],
    ],
    "true_sequential": True,
    "skipped_layers": ["lm_head"],
}

gpt_fast_model = gpt_fast(model_name, calibration_data_function=get_wikitext2, quantize_config=quantize_config, sample_function=argmax, cache_config=cache_config, \
                          device=device, draft_model_name=draft_model_name)
gpt_fast_model.to(device)

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)

print(tokenizer.decode(res[0]))