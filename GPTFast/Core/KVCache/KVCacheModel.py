import types
from typing import Callable
import torch
from torch import nn
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers import AutoModelForCausalLM
from GPTFast.Helpers.Class import *
from GPTFast.Helpers.String import *
from .KVCache import KVCache

INFERENCE_BATCH_SIZE = 1

class KVCacheModel(nn.Module):
    def __init__(self, model:nn.Module, sample_fn:Callable[..., torch.Tensor], cache_config:dict, dtype:torch.dtype, device:torch.device):
        super().__init__()
        self.device = device
        self.sample = types.MethodType(sample_fn, self)
        assert not isinstance(model, BloomForCausalLM), "Bloom models currently have an unsupported kv cache shape."

        self._model = self.add_static_cache_to_model(model, cache_config, dtype, self.device)
        config = self._model.config
        self._max_length = cache_config["max_length"]
        self._num_hidden_layers = config.num_hidden_layers
        self._num_attention_heads = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
        self._head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        self._vocab_size = config.vocab_size
        self._cached_len = 0
        self._init_prob_history = torch.empty((INFERENCE_BATCH_SIZE, 0, self._vocab_size)).to(self.device)
        self._prob_history = self._init_prob_history
        self._full_key_values = torch.zeros((self._num_hidden_layers, 2, INFERENCE_BATCH_SIZE, \
                                            self._num_attention_heads, self._max_length, self._head_dim)).to(self.device)
        self._init_key_values = torch.zeros((self._num_hidden_layers, 2, INFERENCE_BATCH_SIZE, \
                                            self._num_attention_heads, self._cached_len, self._head_dim)).to(self.device)
        self._dummy_key_values = self._init_key_values

    @classmethod
    def add_static_cache_to_model(cls, model:AutoModelForCausalLM, cache_config:dict, dtype:torch.dtype, device:torch.device):
        assert "model_config" in cache_config, "you must specify how your model will be updated to accomadate a static kv cache."
        model_config = cache_config["model_config"]
        assert "path_to_blocks" in model_config, "you must specify how to reach the blocks from the model."
        assert "child_ref_in_parent_forward" in model_config, "you must specify how each child module is specified in the forward method of the parent module."
        imports = cache_config["imports"]

        module_with_input_pos = model
        for (prop, forward_prop_ref) in zip(model_config["path_to_blocks"], model_config["child_ref_in_parent_forward"]):
            module_forward_str = get_method_str(module_with_input_pos, "forward")
            module_forward_str_kv_cache = add_input_pos_to_func_str(module_forward_str, forward_prop_ref, "input_pos=input_pos")
            module_forward_str_kv_cache = add_default_parameter(module_forward_str_kv_cache, "forward", "input_pos", "Optional[torch.Tensor]", None, True)
            add_str_as_func(module_with_input_pos, "forward", module_forward_str_kv_cache, imports)

            module_with_input_pos = getattr(module_with_input_pos, prop)
        
        assert isinstance(module_with_input_pos, nn.ModuleList), "Once we finish iterating through 'path_to_blocks', the property that you arrive at must be a nn.ModuleList."
        assert "block_config" in cache_config, "you must specify how your blocks will be updated to accomadate the static kv cache."

        block_config = cache_config["block_config"]
        assert "path_to_attn" in block_config, "you must specify how to reach the attention layer from the blocks."
        assert "child_ref_in_parent_forward" in block_config, "you must specify how each child module is specified in the forward method of the parent module."

        for block in module_with_input_pos:
            sub_module_with_input_pos = block
            for (prop, forward_prop_ref) in zip(block_config["path_to_attn"], block_config["child_ref_in_parent_forward"]):
                sub_module_forward_str = get_method_str(sub_module_with_input_pos, "forward")
                sub_module_forward_str_kv_cache = add_input_pos_to_func_str(sub_module_forward_str, forward_prop_ref, "input_pos=input_pos")
                sub_module_forward_str_kv_cache = add_default_parameter(sub_module_forward_str_kv_cache, "forward", "input_pos", "Optional[torch.Tensor]", None, True)
                add_str_as_func(sub_module_with_input_pos, "forward", sub_module_forward_str_kv_cache, imports)

                sub_module_with_input_pos = getattr(sub_module_with_input_pos, prop)

            assert "attn_config" in cache_config, "you must specify how the attention layer is modified to accomodate the static kv cache."
            attn_config = cache_config["attn_config"]
            assert "cache_update_config" in attn_config, "you must specify how the static kv cache is updated in your attention layer."
            cache_update_config = attn_config["cache_update_config"]

            attention_layer = sub_module_with_input_pos
            assert "max_length" in cache_config, "You must specify how large your kv cache will be before it can be statically allocated."
            max_length = cache_config["max_length"]
            cache = KVCache(model.config, 1, max_length, device = device, dtype=dtype)  
            attention_layer.kv_cache = cache

            assert "kv_cache_condition" in cache_update_config, "you must specify the condition under which the key-value cache is updated in the attention layer."
            assert "key_name" in cache_update_config, "you must specify a key name to be updated."
            assert "value_name" in cache_update_config, "you must specify a value name to be updated." 
            key_name, value_name = cache_update_config["key_name"], cache_update_config["value_name"]
            new_key_name, new_value_name = cache_update_config.get("new_key_name", key_name), cache_update_config.get("new_value_name", value_name)
            
            #this will add the kv cache to the attention layer.
            attn_forward_str = get_method_str(attention_layer, "forward")
            attn_forward_str_kv_cache = modify_if_block(attn_forward_str, cache_update_config["kv_cache_condition"], \
                [f'{new_key_name}, {new_value_name} = self.kv_cache.update({key_name}, {value_name}, -1, input_pos=input_pos)'])
            attn_forward_str_kv_cache = add_default_parameter(attn_forward_str_kv_cache, "forward", "input_pos", "Optional[torch.Tensor]", None, True)

            assert "causal_mask_config" in attn_config, "you must specify how the causal mask needs to be updated to accomadate the static kv cache."
            causal_mask_config = attn_config["causal_mask_config"]
            assert "causal_mask_application" in causal_mask_config, "you must specify whether the causal mask is applied conditionally or not."
            assert "causal_mask_method" in causal_mask_config, "you must specify the name of the method where the causal mask is applied to QK^T."

            if causal_mask_config["causal_mask_method"] == "forward":
                attn_forward_str_causal_mask = shift_right(attn_forward_str_kv_cache)
            else:
                add_str_as_func(attention_layer, "forward", attn_forward_str_kv_cache, imports)
                attn_forward_str_causal_mask = get_method_str(attention_layer, causal_mask_config["causal_mask_method"])

            if causal_mask_config["causal_mask_application"] == "conditional":
                assert "causal_mask_condition" in causal_mask_config, "you must specify the condition under which the causal mask is applied to QK^T."

                #this will add the causal mask to the _attn method of the attention layer.
                attn_forward_str_causal_mask_modified = modify_if_block(attn_forward_str_causal_mask, causal_mask_config["causal_mask_condition"], \
                                                        ["causal_mask = self.kv_cache.partial_mask.view(attn_weights.shape)", \
                                                        "attn_weights = attn_weights + causal_mask"])
            else:
                assert "causal_mask_line" in causal_mask_config, "you must specify the first line that you want to overwrite to apply the causal mask to QK^T."
                assert "num_lines" in causal_mask_config, "you must specify how many lines you want to overwrite to apply the causal mask to QK^T."

                attn_forward_str_causal_mask_modified = modify_function_block(attn_forward_str_causal_mask, causal_mask_config["causal_mask_line"], causal_mask_config["num_lines"], \
                                                        ["causal_mask = self.kv_cache.partial_mask.view(attn_weights.shape)", \
                                                        "attn_weights = attn_weights + causal_mask"])
                
            add_str_as_func(attention_layer, causal_mask_config["causal_mask_method"], attn_forward_str_causal_mask_modified, imports)

        return model

    @torch.no_grad()
    def prefill(self, input_ids:torch.Tensor) -> torch.Tensor:
        outputs = self._model(input_ids=input_ids, past_key_values=self._dummy_key_values, input_pos=torch.arange(self._cached_len, input_ids.shape[-1]))
        prob_history = outputs.logits
        self._cached_len = input_ids.shape[-1]
        last_q = prob_history[:, -1, :]
        self._prob_history = torch.cat([self._prob_history, prob_history], dim=-2)
        self._dummy_key_values = self._full_key_values[:, :, :, :, :self._cached_len]
        return last_q

    @torch.no_grad()
    def forward(self, input_ids:torch.Tensor) -> torch.Tensor:
        # return the last token's logits
        input_pos = torch.arange(self._cached_len, self._cached_len + input_ids.shape[-1])
        outputs = self._model(input_ids=input_ids, past_key_values=self._dummy_key_values, input_pos=input_pos)
        prob_history = outputs.logits
        self._cached_len = self._cached_len + input_ids.shape[-1]
        last_q = prob_history[:, -1, :]
        self._prob_history = torch.cat([self._prob_history, prob_history], dim=-2)
        self._dummy_key_values = self._full_key_values[:, :, :, :, :self._cached_len]
        return last_q

    @torch.no_grad()
    def generate(self, cur_tokens:torch.Tensor, max_tokens:int, **sampling_kwargs) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = cur_tokens

        #prefill stage
        last_logits = self.prefill(x)
        last_prob = torch.nn.functional.softmax(last_logits, dim=-1)
        next_tok = self.sample(last_prob, **sampling_kwargs)
        x = torch.cat((x, next_tok), dim=1)
        length = max_tokens - len(x[0])

        for _ in range(length):
            last_logits = self.forward(next_tok)
            last_prob = torch.nn.functional.softmax(last_logits, dim=-1)
            next_tok = self.sample(last_prob, **sampling_kwargs)
            x = torch.cat((x, next_tok), dim=1)
        
        self.clear()

        return x

    @torch.no_grad()
    def decode_function(self, uncached_input_ids:torch.Tensor, length:int, return_text:bool=True, **sampling_kwargs) -> torch.Tensor:

        all_input_ids = torch.empty((INFERENCE_BATCH_SIZE, 0), dtype=torch.long).to(self.device)

        for _ in range(length):
            last_logits = self.forward(uncached_input_ids)

            token_probabilities = torch.nn.functional.softmax(last_logits, dim=-1)
            next_token = self.sample(token_probabilities, **sampling_kwargs)
            # Append the sampled token to the input sequence
            uncached_input_ids = next_token
            all_input_ids = torch.cat([all_input_ids, next_token], dim=-1)
        
        all_probabilities = torch.nn.functional.softmax(self._prob_history[:, -length-1:-1, :], dim=-1).squeeze(0)

        if return_text:
            return all_input_ids, all_probabilities
        else:
            return all_probabilities
        
    def rollback_cache(self, last_pos:int) -> None:
        self._cached_len = last_pos
        self._dummy_key_values = self._full_key_values[:, :, :, :, :self._cached_len]
        self._prob_history = self._prob_history[:, :self._cached_len, :]
    
    def clear(self):
        self._dummy_key_values = self._init_key_values
        self._prob_history = self._init_prob_history
        self._cached_len = 0

def add_kv_cache(model:nn.Module, sample_fn:Callable, cache_config:dict, dtype:torch.dtype, device:torch.device) -> KVCacheModel:
    model = KVCacheModel(model, sample_fn, cache_config, dtype, device)
    return model