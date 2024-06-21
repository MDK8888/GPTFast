from enum import Enum
from typing import Tuple, Optional
import torch
from ..Constants import *
from .get_and_check_qmin_qmax import *
from .get_reduction_params import *

def quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
):
    """
    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported input dtype: {input.dtype}"
    quant_min, quant_max = get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    shape_for_reduction, reduction_dims = get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT:
        quant = torch.clamp(
            torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max
        ).to(output_dtype)
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT
        mid_point = (quant_max + quant_min + 1) / 2
        min_val = zero_point - scale * mid_point
        quant = (
            torch.clamp(
                torch.round((input - min_val) / scale),
                quant_min, quant_max)
        ).to(output_dtype)
    quant = quant.view(original_shape)

    return quant