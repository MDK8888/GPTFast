
from typing import Tuple, Optional
import torch
from ..Constants import *
from .get_and_check_qmin_qmax import *
from .get_reduction_params import *


def dequantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    *,
    output_dtype: torch.dtype = torch.float32,
):
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                               e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """

    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype == input_dtype, f"Expected: {input_dtype}, got: {input.dtype}"
    assert output_dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported output dtype: {output_dtype}"
    quant_min, quant_max = get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)

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
        # Force a copy to avoid input modification due
        # to upcoming in-place operations.
        dequant = input.to(torch.int32, copy=True)
        if zero_point is not None:
            dequant -= zero_point.to(torch.int32)
        dequant = dequant.to(output_dtype)
        dequant *= scale
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT, f"Unexpected zero point domain: {zero_point_domain}"
        mid_point = (quant_max + quant_min + 1) / 2
        # This should allocate new memory and avoid input modification
        dequant = input - mid_point
        dequant = dequant.to(output_dtype)
        dequant *= scale
        if zero_point is not None:
            dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)