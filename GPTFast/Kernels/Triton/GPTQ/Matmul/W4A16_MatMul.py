import torch
import triton
import triton.language as tl

@triton.jit
def int4_matmul_kernel_3d_coalesce(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_n, stride_scales_g,
    stride_zeros_n, stride_zeros_g,
    M, N, K,
    groupsize: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Make sure we don't access memory out of bounds
    offs_am = tl.max(offs_am, 0)
    offs_am = tl.min(offs_am, M)
    offs_bn = tl.max(offs_bn, 0)
    offs_bn = tl.min(offs_bn, N)
    offs_k = tl.max(offs_k, 0)
    offs_k = tl.min(offs_k, K)

    # Load a and b
    a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    # Load scales and zeros
    scales = tl.load(scales_ptr + offs_bn[None, :] * stride_scales_n + (offs_k[:, None] // groupsize) * stride_scales_g)
    zeros = tl.load(zeros_ptr + offs_bn[None, :] * stride_zeros_n + (offs_k[:, None] // groupsize) * stride_zeros_g)

    # Unpack int4 values
    b = (b >> ((offs_k % 8)[:, None] * 4)) & 0xF

    # Dequantize
    b = (b.to(tl.float32) - 8) * scales + zeros

    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(a, b)

    # Store output
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def int4_matmul_coalesce(a, b, scales, zeros, groupsize: int = 128):
    # Check input requirements
    assert a.is_cuda and b.is_cuda and scales.is_cuda and zeros.is_cuda
    assert b.shape[0] == a.shape[1] // 8, "Packed weight shape mismatch"
    
    M, K = a.shape
    _, N = b.shape

    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # This can be adjusted based on your specific needs

    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M), 
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(K, BLOCK_K)
    )

    # Initialize output tensor
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Launch kernel
    int4_matmul_kernel_3d[grid](
        a, b, c, scales, zeros,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        M, N, K,
        groupsize,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )

    return c


@triton.jit
def int4_matmul_kernel_3d(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_n, stride_scales_g,
    stride_zeros_n, stride_zeros_g,
    M, N, K,
    groupsize: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_id = tl.program_id(0)
    n_id = tl.program_id(1)
    k_id = tl.program_id(2)  # New dimension in the grid

    offs_am = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_id * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_scales_n + (offs_k[:, None] // groupsize) * stride_scales_g
    zeros_ptrs = zeros_ptr + offs_bn[None, :] * stride_zeros_n + (offs_k[:, None] // groupsize) * stride_zeros_g

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)

    # Unpack int4 values
    mask = tl.arange(0, BLOCK_K)[:, None] % 8
    b = (b >> (mask * 4)) & 0xF

    # Dequantize
    b = (b - 8) * scales + zeros

    acc = tl.dot(a, b)

    # Accumulate results atomically
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc.to(tl.float16))

@torch.compile(fullgraph=True)
def int4_matmul(a, b, scales, zeros, groupsize: int = 128):
    assert a.is_cuda and b.is_cuda and scales.is_cuda and zeros.is_cuda
    assert b.shape[0] == a.shape[1] // 8, "Packed weight shape mismatch"
    
    M, K = a.shape
    _, N = b.shape

    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

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
        groupsize,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=2,
        num_stages=3,
    )

    return c