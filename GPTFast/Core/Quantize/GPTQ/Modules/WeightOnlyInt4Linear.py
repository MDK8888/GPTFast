import torch
import torch.nn as nn
from typing import Union
from GPTFast.Kernels import int4_matmul

class WeightOnlyInt4Linear(nn.Module):
    def __init__(self, in_features: int, 
                       out_features: int, 
                       name: str, 
                       weight: torch.IntTensor,
                       scales: torch.Tensor,
                       zeros: torch.Tensor,
                       bias: Union[torch.Tensor, None] = None,
                       groupsize: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.name = name

        self.weight = weight
        assert self.weight.shape == (in_features // 8, out_features), f"Your weight needs to be of shape ({in_features // 8}, {out_features}), but it currently has shape {self.weight.shape}"
        assert torch.isnan(self.weight).any().item() == False, "Your self.weight contains NaN."

        self.scales = scales
        assert self.scales.shape == (out_features, in_features // groupsize), f"Your scales needs to be of shape ({out_features}, {in_features // 8}), but it currently has shape {self.scales.shape}"
        assert torch.isnan(self.scales).any().item() == False, "Your self.scales contains NaN."

        self.zeros = zeros
        assert self.zeros.shape == (out_features, in_features // groupsize), f"Your zeros needs to be of shape ({out_features}, {in_features // 8}), but it currently has shape {self.scales.shape}"
        assert torch.isnan(self.zeros).any().item() == False, "Your self.zeros contains NaN."

        if bias is not None:
            self.bias = bias
        else:
            self.bias = torch.zeros(out_features, device=weight.device, dtype=torch.float16)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_dtype = input.dtype
        reshaped_input = input.to(torch.float16).view(-1, self.in_features)
        output = int4_matmul(reshaped_input, self.weight, self.scales, self.zeros, self.groupsize)
        new_output_shape = input.shape[:-1] + (output.shape[-1],)
        output = output.view(new_output_shape) + self.bias
        return output.to(orig_dtype)