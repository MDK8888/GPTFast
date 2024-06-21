import os
import math
from typing import Union
import torch
from torch import nn
import transformers
from ..Constants import *
from ..Functions import *
from ...Quantizer import Quantizer

class GPTQLinearModuleQuantizer(Quantizer):

    def __init__(self, layer:Union[nn.Linear, transformers.Conv1D], blocksize:int = 128, percdamp:float = 0.01, groupsize:int = 128, device:torch.device = "cpu"):
        self.layer = layer
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.device = device
        self.nsamples = 0

        W = layer.weight.data.clone()
        if isinstance(layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns)).to(self.device)

    def get_qparams_func(self, w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
        if groupsize > w.shape[-1]:
            groupsize = w.shape[-1]
        assert groupsize > 1
        assert w.shape[-1] % groupsize == 0
        assert w.dim() == 2
        assert n_bit <= 8, f"only n_bit smaller than 8 is supported, got: {n_bit}"

        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int32
        block_size = (1, groupsize)
        quant_min = 0
        quant_max = 2**n_bit - 1
        eps = 1e-6
        scale_dtype = dtype
        zero_point_dtype = dtype

        scale, zero_point = choose_qparams_affine(
            w,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            scale_dtype=scale_dtype,
            zero_point_dtype=zero_point_dtype,
            preserve_zero=False,
            zero_point_domain=ZeroPointDomain.FLOAT
        )

        return scale.to(dtype=dtype).reshape(w.shape[0], -1), zero_point.to(
            dtype=dtype
        ).reshape(w.shape[0], -1)

    #ok, so basically this function returns to us the scales and zeros. What does this mean?
    #Essentially, in quantization, what's going on is we have a weight W. In order to quantize it 
    #to n bits, we need to send it to a range representable by n bits with (W - min_val)/scales.
    #This function gets us our zeros and scales which are needed for the above.
    def quantize_func(
        self,
        w,
        scales,
        zeros,
        n_bit=4,
        groupsize=128,
    ):
        assert groupsize > 1
        # needed for GPTQ single column quantize
        if groupsize > w.shape[-1] and scales.shape[-1] == 1:
            groupsize = w.shape[-1]

        assert w.shape[-1] % groupsize == 0, "The shape of your weights doesn't divide your groupsize. Try decreasing the groupsize."
        assert w.dim() == 2

        block_size = (1, groupsize)
        output_dtype = torch.int32
        quant_min = 0
        quant_max = 2 ** n_bit - 1

        return quantize_affine(w, block_size, scales, zeros, output_dtype, quant_min, quant_max, zero_point_domain = ZeroPointDomain.FLOAT)
    
    def dequantize_func(
        self,
        w_int4x8,
        scales,
        zeros,
        n_bit=4,
        groupsize=128,
    ):
        assert groupsize > 1
        # needed for GPTQ single column dequantize
        if groupsize > w_int4x8.shape[-1] and scales.shape[-1] == 1:
            groupsize = w_int4x8.shape[-1]
        assert w_int4x8.shape[-1] % groupsize == 0
        assert w_int4x8.dim() == 2

        block_size = (1, groupsize)
        input_dtype = torch.int32
        quant_min = 0
        quant_max = 2**n_bit - 1
        return dequantize_affine(w_int4x8, block_size, scales, zeros, input_dtype, quant_min, quant_max, zero_point_domain=ZeroPointDomain.FLOAT, output_dtype=scales.dtype)
    
    def combine_qparams_list_func(self, qparams_list):
        result = []
        for x in zip(*qparams_list):
            concatenated = torch.cat(x, dim=1)
            result.append(concatenated)
        return result

    def add_batch(self, inp:torch.Tensor, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def quantize(self) -> tuple[torch.Tensor]:
        H = self.H
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        percdamp = self.percdamp
        blocksize = self.blocksize
        groupsize = self.groupsize
        orig_dtype = W.dtype
        W = W.detach().float()
        _, columns = W.shape[0], W.shape[1]
        device = W.device

        if groupsize == -1:
            cur_qparams = self.get_qparams_func(W)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        DQ = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        all_qparams = []
        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            DQ1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:  # start of new group
                    cur_qparams = self.get_qparams_func(
                        W[:, (i1 + i) : (i1 + i + groupsize)], groupsize=groupsize
                    )
                    all_qparams.append(cur_qparams)

                scales, zeros = cur_qparams

                q = self.quantize_func(w.unsqueeze(1), scales=scales, zeros=zeros, groupsize=groupsize).flatten()

                #  `dequantize_func`.

                dq = self.dequantize_func(q.unsqueeze(1), scales=scales, zeros=zeros, groupsize=groupsize).flatten()

                DQ1[:, i] = dq
                Losses1[:, i] = (w - dq) ** 2 / d**2

                err1 = (w - dq) / d
                W1[:, i:] -= (
                    err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                )
                Err1[:, i] = err1

            DQ[:, i1:i2] = DQ1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if all_qparams == []:

            all_qparams.append(cur_qparams)

        # convert a list of qparams objects into a single one. enerally by
        # concatenating a bunch of n,1 scale/zeros tensors into a n,num_groups tensor

        #  `combine_qparams_list_func`.
        all_qparams = self.combine_qparams_list_func(all_qparams)
        scales, zeros = all_qparams
        Q = self.quantize_func(DQ, scales=scales, zeros=zeros, groupsize=groupsize)
        return Q, DQ.to(orig_dtype), all_qparams

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

def find_layers_dict(module, layers=None, name=""):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers_dict(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res

def pack_tinygemm_scales_and_zeros(scales, zeros):
    guard_dtype_size(scales, "scales", dtype=torch.bfloat16, size=zeros.size())
    guard_dtype_size(zeros, "zeros", dtype=torch.bfloat16)
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

def guard_dtype_size(tensor_arg, arg_name, dtype=None, size=None):
    if dtype is not None and tensor_arg.dtype != dtype:
        raise ValueError("Expected Tensor argument {arg_name} to have dtype {dtype}, but got {tensor_arg.dtype} instead.")
    if size is not None and tensor_arg.size() != size:
        raise ValueError("Expected Tensor argument {arg_name} to have size {size}, but got {tensor_arg.size()} instead.")