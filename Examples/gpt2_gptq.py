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

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
initial_string = "Hello my name is"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

def prepare_calibration_data(self, num_samples=24, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    calibration_data = []
    
    for item in dataset.take(num_samples):
        tokens = tokenizer.encode(item['text'], max_length=max_length, truncation=True, return_tensors="pt")
        calibration_data.append(tokens)
    
    return calibration_data

quantize_config = {
    "quantization_mode": "GPTQ",
    "save_quantized_state_dict": True,
    "groupsize": 32,
    "percdamp": 0.1,
    "path_to_blocks": ["transformer", "h"],
    "inside_layer_modules": [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"]
    ],
    "true_sequential": True,
    "skip_layers": ["lm_head"],
    "bias": True,
    "log_quant_stats": True
}

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

N_ITERS=10
MAX_TOKENS=50

gpt_fast_model = gpt_fast(model_name, calibration_data_function=prepare_calibration_data, quantize_config=quantize_config, sample_function=argmax, cache_config=cache_config, \
                          device=device)
gpt_fast_model = gpt_fast_model.to(device)

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        res, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)

print(tokenizer.decode(res[0]))