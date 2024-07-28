import os
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from GPTFast.Core import gpt_fast
from GPTFast.Helpers import timed

torch._dynamo.reset()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index.view(1, 1)

model_name = "gpt2-xl"
draft_model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
initial_string = "Hello my name is"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

cache_config = {
    "max_length": 60,
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
                "from torch import LongTensor",
                "import transformers", 
                "from transformers.pytorch_utils import *",
                "from transformers.modeling_utils import *",
                "from transformers.modeling_outputs import *",
                "from transformers.utils import *",
                "from typing import *", 
                "import types"
                ]
}

def get_wikitext2(self):

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

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
    "path_to_blocks": ["transformer", "h"],
    "inside_layer_modules": [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"]
    ],
    "true_sequential": True,
    "skipped_layers": ["lm_head"],
}
gpt_fast_model = gpt_fast(model_name, calibration_data_function=get_wikitext2, quantize_config=quantize_config, sample_function=argmax, cache_config=cache_config, \
                          device=device, profile=False, draft_model_name=draft_model_name)
gpt_fast_model = gpt_fast_model.to(device)

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)

print(tokenizer.decode(res[0]))

gpt_fast_model.profile_enabled = True

gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=6)