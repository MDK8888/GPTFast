import torch
from torch import nn
from ..Cache import NativeCache

class NativeCacheWrapper(nn.Module):

    def __init__(self, model, cache_config, max_generated_length:int, dtype:torch.dtype, device:torch.device):
        super(NativeCacheWrapper, self).__init__()
        self._model = self.add_static_cache_to_model(model, cache_config, max_generated_length, dtype, device)
        self.config = model.config
        self.cache_config = cache_config
        self.cache_position_kwargs = {cache_config["cache_position_arg_name"]: None}

    @classmethod
    def add_static_cache_to_model(self, model:nn.Module, cache_config:dict, max_generated_length:int, dtype:torch.dtype, device:torch.device):
        assert "model_config" in cache_config, "you must specify a model_config to activate static caching for a model which supports it natively."
        model_config = cache_config["model_config"]
        assert "path_to_blocks" in model_config, "you must specify how to access the blocks to activate static caching for a model which supports it natively."
        path_to_blocks = model_config["path_to_blocks"]
        cur_module = model
        for attr in path_to_blocks:
            cur_module = getattr(cur_module, attr)
        assert isinstance(cur_module, nn.ModuleList), "your path_to_blocks does not terminate in an instance of nn.ModuleList."
        blocks = cur_module
        assert "block_config" in cache_config, "you must specify how a block_config to activate static caching for a model which supports it natively."
        block_config = cache_config["block_config"]
        assert "path_to_attn" in block_config, "you must specify how to access the attention layer from the block to activate static caching for a model which supports it natively."
        path_to_attn = block_config["path_to_attn"]
        for block in blocks:
            cur_module = block
            for attr in path_to_attn:
                cur_module = getattr(cur_module, attr)
            attn = cur_module
            attn.past_key_value = NativeCache(model.config, 1, max_generated_length, device, dtype)
        return model

    def forward(self, input_ids:torch.LongTensor, past_key_values:torch.Tensor, input_pos:torch.LongTensor):
        self.cache_position_kwargs[self.cache_config["cache_position_arg_name"]] = input_pos #this line may cause performance difficulties
        output = self._model(input_ids=input_ids, use_cache=True, **self.cache_position_kwargs)
        return output