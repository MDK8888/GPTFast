import torch
import triton
import triton.language as tl

torch.set_float32_matmul_precision('high')

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N":32, "BLOCK_K":64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N":32, "BLOCK_K":64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N":32, "BLOCK_K":64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N":32, "BLOCK_K":64}, num_stages=3, num_warps=4),
    ],
    key=[],
)

@triton.jit
def int4_matmul_kernel_3d(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_n, stride_scales_g,
    stride_zeros_n, stride_zeros_g,
    M, N, K,
    groupsize,
    BLOCK_M:tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_id = tl.program_id(0)
    n_id = tl.program_id(1)
    k_id = tl.program_id(2)  # New dimension in the grid
    
    offs_am = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_id * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_am < M

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_scales_n + (offs_k[:, None] // groupsize) * stride_scales_g
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_zeros_n + (offs_k[:, None] // groupsize) * stride_zeros_g

    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
    b = tl.load(b_ptrs)

    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)

    # Unpack int4 values
    mask = tl.arange(0, BLOCK_K)[:, None] % 8
    b = (b >> (mask * 4)) & 0xF

    # Dequantize
    b = (b - 8) * scales + zeros

    acc = tl.dot(a, b.to(tl.float16))

    # Accumulate results atomically
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc.to(tl.float16))

@torch.compile(fullgraph=True)
def int4_matmul(a:torch.Tensor, b:torch.Tensor, scales:torch.Tensor, zeros:torch.Tensor, groupsize: int = 128):
    assert a.is_cuda and b.is_cuda and scales.is_cuda and zeros.is_cuda
    assert b.shape[0] == a.shape[1] // 8, "Packed weight shape mismatch"

    M, K = a.shape
    _, N = b.shape

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))

    # Initialize output tensor with zeros
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)

    int4_matmul_kernel_3d[grid](
        a, b, c, scales, zeros,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        M, N, K,
        groupsize
    )

    return c