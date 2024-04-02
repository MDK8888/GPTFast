import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GPTFast.Core import KVCacheModel

initial_string = "Write me a short story."

model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_tokens = tokenizer.encode(initial_string, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_name)

outputs = model(input_ids=input_tokens)

print(outputs.logits)

config = {
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
    "imports": ["import torch", 
                "import transformers", 
                "from transformers import *", 
                "from torch import *", 
                "from typing import *", 
                "import types", 
                "from transformers.modeling_outputs import *", 
                "from torch import nn", 
                "from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask"]
}

model = KVCacheModel.add_static_cache_to_model(model, cache_config=config, max_generated_length=60, dtype=torch.float32, device="cpu")

config = model.config
name = config.name_or_path
max_length = 7
num_hidden_layers = config.num_hidden_layers
num_attention_heads = config.num_attention_heads
hidden_size = config.hidden_size
vocab_size = config.vocab_size

dummy_key_value = torch.ones(num_hidden_layers, 2, 1, num_attention_heads, 0, hidden_size // num_attention_heads, dtype=torch.int8)
outputs = model(input_ids=input_tokens, past_key_values=dummy_key_value, input_pos=torch.arange(0, 7))

print(outputs.logits)