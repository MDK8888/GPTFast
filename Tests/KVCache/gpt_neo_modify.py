import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core import KVCacheModel

model_name = "EleutherAI/gpt-neo-125m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Write me a short story."
prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")

output = model(input_ids=prompt_tokens, use_cache=True)

print(output.logits)
past_key_values = output.past_key_values

response = "It was a dark and stormy night."
response_tokens = tokenizer.encode(response, return_tensors="pt")

output = model(input_ids=response_tokens, past_key_values=past_key_values)

print(output.logits)

config = {
    "model_config": {
        "path_to_blocks": ["transformer", "h"],
        "child_ref_in_parent_forward": ["transformer", "block"],
    },
    "block_config": {
        "path_to_attn": ["attn", "attention"],
        "child_ref_in_parent_forward": ["attn", "attention"], 
    },
    "attn_config": {
        "cache_update_config":{
            "kv_cache_condition":"if layer_past is not None",
            "key_name": "key",
            "value_name": "value",
        },
        "causal_mask_config": {
            "causal_mask_application": "",
            "causal_mask_method": "_attn",
            "causal_mask_line": "query_length, key_length = query.size(-2), key.size(-2)",
            "num_lines": 11
        }
    },
    "imports": ["import torch", 
                "import transformers", 
                "from transformers import *", 
                "from torch import *", 
                "from typing import *", 
                "import types", 
                "from transformers.modeling_outputs import *", 
                "from torch import nn", 
                "from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask"
                ]
}

model = KVCacheModel.add_static_cache_to_model(model, cache_config=config, max_generated_length=100, dtype=torch.float32, device="cpu")

config = model.config
name = config.name_or_path
max_length = 6
num_hidden_layers = config.num_hidden_layers
num_attention_heads = config.num_attention_heads
hidden_size = config.hidden_size
vocab_size = config.vocab_size

dummy_key_value = torch.ones(num_hidden_layers, 2, 1, num_attention_heads, 0, hidden_size // num_attention_heads, dtype=torch.int8)

outputs = model(input_ids=prompt_tokens, past_key_values=dummy_key_value, input_pos=torch.arange(0, 6))

print(outputs.logits)

dummy_key_value = torch.ones(num_hidden_layers, 2, 1, num_attention_heads, 6, hidden_size // num_attention_heads, dtype=torch.int8)

outputs = model(input_ids=response_tokens, past_key_values=dummy_key_value, input_pos=torch.arange(6, 6 + response_tokens.shape[-1]))

print(outputs.logits)
