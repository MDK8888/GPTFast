from typing import Optional, Tuple, Dict, Any
import torch
from transformers.cache_utils import Cache

class KVCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `max_position_embeddings`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        min_value = torch.finfo(self.dtype).min
        self.full_mask = torch.zeros((max_batch_size, self.num_key_value_heads, self.max_cache_len, self.max_cache_len), dtype=self.dtype, device=device)
        min_indices = torch.triu(torch.ones((max_batch_size, self.num_key_value_heads, self.max_cache_len, self.max_cache_len), dtype=torch.bool), diagonal=1)
        self.full_mask[min_indices] = min_value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        full: Optional[bool] = False,
        **cache_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for. Kept for backward compatibility
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` just needs the `q_len`
                to know how much of the cache it should overwrite.

        Return:
            A tuple containing the updated key and value states.
        """
        new_cache_positions = cache_kwargs.get("input_pos")
        k_out = self.key_cache
        v_out = self.value_cache

        k_out[:, :, new_cache_positions] = key_states
        v_out[:, :, new_cache_positions] = value_states

        self.partial_mask = self.full_mask[:, :, new_cache_positions]
        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC"""
        return self.seen_tokens

    def get_usable_length(self, new_sequence_length=None, layer_idx: Optional[int] = 0) -> int:
        return self.seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self):
        """Dummy func for BC. We have to keep it because otherwise the call in the forward of models will break it"""
        return None