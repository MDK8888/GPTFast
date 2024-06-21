import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core import KVCacheModel

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    initial_string = "Write me a short story."
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    initial_output = model(input_tokens)

    print(initial_output.logits)

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

    config = model.config
    name = config.name_or_path
    max_length = 6
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    dummy_key_value = torch.ones(num_hidden_layers, 2, 1, num_attention_heads, 0, hidden_size // num_attention_heads, dtype=torch.int8)

    outputs = model(input_ids=input_tokens, past_key_values=dummy_key_value, input_pos=torch.arange(0, 6))

    print(outputs.logits)

    dummy_key_value = torch.ones(num_hidden_layers, 2, 1, num_attention_heads, 6, hidden_size // num_attention_heads, dtype=torch.int8)

    response = "It was a dark and story night."

    response_tokens = tokenizer.encode(response, return_tensors="pt")

    outputs = model(input_ids=response_tokens, past_key_values=dummy_key_value, input_pos=torch.arange(6, 14))

    print(outputs.logits)