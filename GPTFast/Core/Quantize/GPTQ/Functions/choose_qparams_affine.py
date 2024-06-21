import torch
from typing import Tuple, Optional
from ..Constants import *
from .get_and_check_qmin_qmax import *
from .get_reduction_params import *

def choose_qparams_affine(
   input: torch.Tensor,
   mapping_type: MappingType,
   block_size: Tuple[int, ...],
   target_dtype: torch.dtype,
   quant_min: Optional[int] = None,
   quant_max: Optional[int] = None,
   eps: Optional[float] = None,
   scale_dtype: Optional[torch.dtype] = None,
   zero_point_dtype: Optional[torch.dtype] = None,
   preserve_zero: bool = True,
   zero_point_domain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        input (torch.Tensor): fp32, bf16, fp16 input Tensor
        mapping_type (MappingType): determines how the qparams are calculated, symmetric or asymmetric
        block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
          e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
        target_dtype (torch.dtype): dtype for target quantized Tensor
        quant_min (Optional[int]): minimum quantized value for target quantized Tensor
        quant_max (Optioanl[int]): maximum quantized value for target quantized Tensor
        eps (Optional[float]): minimum scale, if not provided, default to eps of input.dtype
        scale_dtype (torch.dtype): dtype for scale Tensor
        zero_point_dtype (torch.dtype): dtype for zero_point Tensor
        preserve_zero (bool): a flag to indicate whether we need zero to be exactly
          representable or not, this is typically required for ops that needs zero padding, like convolution
          it's less important for ops that doesn't have zero padding in the op itself, like linear.

          For example, given a floating point Tensor [1.2, 0.1, 3.0, 4.0, 0.4, 0], if `preserve_zero` is True,
          we'll make sure there is a integer value corresponding to the floating point 0, e.g. [-3, -8, 3, 7, -7, -8], 0 will be mapped to `-8` without loss. But if `preserve_zero` is not True, there won't be such
          gurantee.

          If we don't need zero to be exactly representable, we won't do rounding and clamping for zero_point

        zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
            if zero_point is in integer domain, zero point is added to the quantized integer value during
            quantization
            if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
            value during quantization
            default is ZeroPointDomain.INT

    Output:
        Tuple of scales and zero_points Tensor with requested dtype
    """
    quant_min, quant_max = get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC], f"Unsupported mapping type: {mapping_type}"

    if scale_dtype is None:
        scale_dtype = input.dtype
    if zero_point_dtype is None:
        zero_point_dtype = input.dtype

    assert len(block_size) == input.dim()
    shape_for_reduction, reduction_dims = get_reduction_params(block_size, input.size())
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=False)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=False)

    if preserve_zero:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    else:
        min_val_neg = min_val
        max_val_pos = max_val

    if mapping_type == MappingType.SYMMETRIC:
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        if not preserve_zero:
            raise ValueError("preserve_zero == False is not supported for symmetric quantization")
        if zero_point_domain != ZeroPointDomain.INT:
            raise ValueError("zero_point_domain != ZeroPointDomain.INT is not supported for symmetric quantization")
        zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        if preserve_zero:
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        else:
            assert zero_point_domain == ZeroPointDomain.FLOAT, "if not preserve_zero, zero_point must be in FLOAT domain"
            mid_point = (quant_max + quant_min + 1) / 2
            zero_point = min_val_neg + scale * mid_point

    if eps is None:
        eps = torch.finfo(input.dtype).eps
    scale = torch.clamp(scale, min=eps)

    return scale.to(dtype=scale_dtype), zero_point.to(dtype=zero_point_dtype)