import types
import torch
import torch.nn as nn
from ..KVCache.KVCacheModel import KVCacheModel

#ok, here's the key behind speculative decoding. We have two models, Mq the small model and Mp the large model. 
#1. Run Mq on prefix and obtain the distribution for x1 q(x).
#2. Run Mp on prefix and prefix + x1 concurrently to get the distributions for x2 and p(x).
#3. If x1 is rejected by Mp, reject and resample x1 from an altered distribution, otherwise keep x1 and x2. 

def speculative_decode_eager(
    self,
    cur_tokens:torch.Tensor,
    speculate_k:int,
    **kwargs
) -> torch.Tensor:

    device = cur_tokens.device

    draft_model_sampling_kwargs = kwargs.get("draft_model_decoding_kwargs", {})

    decode_input = cur_tokens

    assert hasattr(self, "draft_model"), "You did not prepare your model properly for speculative decoding. Make sure that you add a draft model."
    draft_model = self.draft_model
    
    draft_tokens, draft_prob = draft_model.decode_function(input_ids=decode_input, length=speculate_k, **draft_model_sampling_kwargs)

    assert len(draft_tokens.shape) == 2 and len(draft_prob.shape) == 2, "Your draft tokens must have shape (1, seq_len) and draft_prob must have shape (seq_len, vocab_size)."

    model_sampling_kwargs = kwargs.get("model_forward_kwargs", {})
    full_tokens = torch.cat([decode_input, draft_tokens], dim=-1).to(device)
    with torch.no_grad():
        model_logits = self.forward(full_tokens, **model_sampling_kwargs).logits
    model_logits = model_logits.squeeze(0)[-draft_tokens.shape[1]-1:, :]
    model_prob = torch.nn.functional.softmax(model_logits, dim=-1)
    
    assert len(model_prob.shape) == 2, "Your model_prob must have shape (seq_len, vocab_size)."

    assert len(model_prob) == len(draft_prob), "In order for speculative decoding to work, the main model must generate the same number of tokens as the draft model."

    p = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

    ratio = p / q
    rand = torch.rand_like(ratio)

    reject_locations = (rand > ratio).nonzero(as_tuple=True)[0]

    if reject_locations.shape[0] != 0:
        n = reject_locations[0].item()
        p = draft_prob[n]
        q = model_prob[n]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        last_token = torch.Tensor([self.sample(new)]).to(device)
        
        return torch.cat([draft_tokens[:, :n], last_token.unsqueeze(0)], dim=-1).long()
    else: #we accept all tokens from the draft model
        last_token = torch.Tensor([self.sample(model_prob[-1])]).to(device)
        return torch.cat([draft_tokens, last_token.unsqueeze(0)], dim=-1).long()

def speculative_decode_kv_cache(
    self,
    uncached_tokens:torch.Tensor,
    speculate_k:int,
    **sampling_kwargs
) -> torch.Tensor:

    assert isinstance(self, KVCacheModel), "Your model must be a KVCache model in order to call speculative_decode_kv_cache."

    device = uncached_tokens.device

    assert hasattr(self, "draft_model"), "You did not prepare your model properly for speculative decoding. Make sure that you add a draft model."
    draft_model = self.draft_model
    
    assert isinstance(draft_model, KVCacheModel), "Your draft model muut be a KVCache model in order to call speculative_decode_kv_cache."

    draft_tokens, draft_prob = draft_model.decode_function(uncached_input_ids=uncached_tokens, length=speculate_k, return_text=True)

    assert len(draft_tokens.shape) == 2 and len(draft_prob.shape) == 2, "Your draft tokens must have shape (1, seq_len) and draft_prob must have shape (seq_len, vocab_size)."

    full_tokens = torch.cat([uncached_tokens, draft_tokens], dim=-1).to(device)
    with torch.no_grad():
        self.forward(full_tokens)
    model_logits = self._prob_history[:, -speculate_k-1:, :].squeeze(0)
    model_prob = torch.nn.functional.softmax(model_logits, dim=-1)

    p = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

    ratio = p / q
    rand = torch.rand_like(ratio)

    reject_locations = (rand > ratio).nonzero(as_tuple=True)[0]

    if reject_locations.shape[0] != 0:
        n = reject_locations[0].item()
        p = draft_prob[n]
        q = model_prob[n]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        last_token = self.sample(new, **sampling_kwargs).to(device)
        self.rollback_cache(self._cached_len - (full_tokens.shape[-1] - n) + 1)
        draft_model.rollback_cache(draft_model._cached_len - (draft_tokens.shape[-1] - n) + 1)
        return torch.cat([draft_tokens[:, :n], last_token], dim=-1).long() #maybe? This is what my gut says.
    else: #we accept all tokens from the draft model
        last_token = self.sample(model_prob[-1], **sampling_kwargs).to(device)
        #self.rollback_cache(n + len(cur_tokens) + 1)
        draft_model(input_ids=draft_tokens[:, -1].view(1, 1))
        #draft_model.rollback_cache(draft_model._cached_len + 1) #This is a little bit suspect, might have to change this - why are we rejecting the last token when it works?
        #assume that draft_model already has a kv cache attached.
        return torch.cat([draft_tokens, last_token], dim=-1).long() #intuitively, this makes sense.

def generate(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, **sampling_kwargs) -> torch.Tensor:

    assert len(cur_tokens.shape) == 2 and cur_tokens.shape[0] == 1, "Your batch size must be 1"

    assert hasattr(self, "speculative_decode"), "You must attach speculative decoding as a method of the model"

    while len(cur_tokens[0]) < max_tokens:
        new_tokens = self.speculative_decode(cur_tokens, speculate_k, **sampling_kwargs)
        cur_tokens = torch.cat((cur_tokens, new_tokens), dim=1).to(torch.long)

    return cur_tokens

def generate_kv_cache(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, **sampling_kwargs) -> torch.Tensor:
    assert isinstance(self, KVCacheModel), "Your model must be a KVCache model for this to work."
    assert hasattr(self, "draft_model") and isinstance(self.draft_model, KVCacheModel), "You must have a draft model and it must be a KVCacheModel for the generate() method."
    assert len(cur_tokens.shape) == 2 and cur_tokens.shape[0] == 1, "Your batch size must be 1"
    assert hasattr(self, "speculative_decode"), "You must attach speculative decoding as a method of the model"

    device = cur_tokens.device

    #prefill phase
    self.draft_model.prefill(cur_tokens)
    new_q = self.prefill(cur_tokens)

    new_token = self.sample(new_q, **sampling_kwargs).to(device)

    cur_tokens = torch.cat((cur_tokens, new_token), dim=-1).to(torch.long)

    while len(cur_tokens[0]) < max_tokens:
        uncached_tokens = cur_tokens[:, -1].view(1, 1)
        new_tokens = self.speculative_decode(uncached_tokens, speculate_k, **sampling_kwargs)
        cur_tokens = torch.cat((cur_tokens, new_tokens), dim=-1).to(torch.long)

    self.clear()
    self.draft_model.clear()

    return cur_tokens    

def add_speculative_decoding(model:nn.Module, draft_model:nn.Module) -> nn.Module:
    model.draft_model = draft_model

    model.speculative_decode = types.MethodType(speculative_decode_kv_cache, model)
    model.generate = types.MethodType(generate_kv_cache, model)
    return model