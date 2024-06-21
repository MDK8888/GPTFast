import torch
from torch import nn
from GPTFast.Helpers import *
from ..Functions import *
import torch.nn.functional as F

class WeightOnlyInt4Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int,
        bias=False, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8,
    ) -> None:
        super().__init__()
        self.padding = not check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        #for Conv1D, we need to transpose everything.
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )