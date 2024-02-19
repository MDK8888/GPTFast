import torch
from torch import nn
from typing import Optional, Callable
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel(nn.Module):
    def __init__(self, model : torch.nn.Module) -> None:
        super().__init__()
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._vocab_size = self._model.config.vocab_size

    @torch.no_grad()
    def prefill(self, input_ids:torch.Tensor) -> torch.Tensor:
        assert self._past_key_values is None and self._prob_history is None
        outputs = self._model(input_ids)
        self._prob_history = outputs.logits #(batch_size, seq_len, vocab_size)
        self._past_key_values = outputs.past_key_values
        last_q = self._prob_history[:, -1, :]
        return last_q

    @torch.no_grad()
    def forward(self, input_ids:torch.Tensor) -> torch.Tensor:
        # return the last token's logits
        k, v = self._past_key_values[-1]
        cached_len = k.shape[2]
            
        last_input_id = input_ids[:, cached_len:]
        
        outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

        not_cached_q = outputs.logits
            
        self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
        
        last_q = not_cached_q[:, -1, :] #(batch_size, 1, vocab_size)
        self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self,prefix:torch.Tensor, 
                                    gamma:int, 
                                    sample:Callable,
                                    **sampling_kwargs) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        #prefill stage
        q = self.prefill(x)
        next_tok = sample(q, **sampling_kwargs)
        x = torch.cat((x, next_tok), dim=1)

        for _ in range(gamma):
            q = self.forward(x)
            next_tok = sample(q, **sampling_kwargs)
            x = torch.cat((x, next_tok), dim=1)
        
        self.clear()
        return x

    def decode_function(self, input_ids:torch.Tensor, length:int, return_text:bool=True) -> torch.Tensor:

        for step in range(length):
            with torch.no_grad():
                q = self.forward(input_ids)

            #print("prob history shape:", self._prob_history.shape)
            token_probabilities = torch.nn.functional.softmax(q, dim=-1)
            max_prob_index = torch.argmax(token_probabilities, dim=-1)
            next_token_id = max_prob_index

            # Append the sampled token to the input sequence
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)
        
        all_probabilities = torch.nn.functional.softmax(self._prob_history[:, -length-1:-1, :], dim=-1).squeeze(0)

        if return_text:
            return input_ids[:, -length:], all_probabilities
        else:
            return all_probabilities

    @torch.no_grad()
    def generate(self, input:torch.Tensor, gamma:int, sample:Callable, **sampling_kwargs) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, sample, **sampling_kwargs)
        return output
    
    @torch.no_grad()
    def rollback_cache(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

    def clear(self):
        self._past_key_values = None
        self._prob_history = None

def add_kv_cache(transformer:nn.Module):
    model = KVCacheModel(transformer)
    return model

