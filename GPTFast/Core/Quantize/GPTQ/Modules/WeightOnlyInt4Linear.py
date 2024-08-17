
import torch
import torch.nn as nn
from GPTFast.Kernels import int4_matmul, pack_int4_weights

class WeightOnlyInt4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, groupsize: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize

        # Initialize quantization parameters
        self.register_buffer('packed_weight', torch.zeros((in_features // 8, out_features), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((out_features, in_features // groupsize), dtype=torch.float32))
        self.register_buffer('zeros', torch.zeros((out_features, in_features // groupsize), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = int4_matmul(input, self.packed_weight, self.scales, self.zeros, self.groupsize)
        if self.bias is not None:
            output += self.bias
        return output