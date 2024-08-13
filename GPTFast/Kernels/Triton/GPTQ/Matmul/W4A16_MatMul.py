import triton
from triton import language as tl
# from actual_base_gptq_4 import triton_matmul4

Q_MIN = 0
Q_MAX = 15
MID = (Q_MIN + Q_MAX + 1) // 2 

@triton.jit()
def swizzle_tile(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n

#we need to rewrite this kernel to take into account the following:
#1. weights are packed horizontally, i.e. if the unpacked quantized weights have size (In, Out), then the packed quantized weights have size (In, Out//8).
#2. Every groupsize columns share scales and zeros. 
#3. Dequantization needs to be done differently.
#4. bfloat16 instead of float16 for numerical stability.

@triton.jit()
def matmul_split_k_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_scales_g, stride_scales_n,
            stride_zeros_g, stride_zeros_n,
            groupsize,
            m, n, k,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            group_m: tl.constexpr, split_k: tl.constexpr):
    
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    total_blocks_k = tl.cdiv(k, block_k*split_k)

    pid_m, pid_n = swizzle_tile(pid,
                                m, n,
                                block_m, block_n, group_m)
    
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // 8) * stride_bn) #b_ptr is now packed horizontally, not vertically.
    #you sneaky bitch. basically, because of offs_bn[None, :] // 8, we have a lot of repeat columns, which is why the bit shift is ok.

    #in our scenario, in the original matrix, scales and zeros both had dimensions (In, Out // groupsize).
    #how does our two lines below need to change to reflect this? Here, scales_ptr, zeros_ptr have sizes (Out // groupsize, In), so they are transposed.

    scales_ptrs = scales_ptr + (offs_bn // groupsize) * stride_scales_n
    zeros_ptrs = zeros_ptr + (offs_bn // groupsize) * stride_zeros_n

    #at this point, scales_ptrs, zeros_ptrs are arrays of pointers to the first elements of the conceptual columns for the scales and 
    #zeros needed for quantization in the first iteration of the loop.

    shifter = (offs_bn % 8) * 4
    #we remove zeros_shifter, not necessary
    #zeros_shifter = (offs_bn % 8) * 4
    
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        #ok, we now need to fix this line. Because we are quantizing horizontally now, get rid of denominator.
        start_id = k * split_k + pid_k
        g_id = start_id * block_k + tl.arange(0, block_k)

        #ok, let's think about ptr right now. ptr is still an array, and it is an array of pointers to the first scales that we need.
        #from this point, we need to load block_k elements vertically - how can we do this?
        ptr = scales_ptrs + g_id[None, :] * stride_scales_g
        scales = tl.load(ptr)
        
        ptr = zeros_ptrs + g_id[None, :] * stride_zeros_g
        zeros = tl.load(ptr) 

        b = (b >> shifter[None, :]) & 0xF
        b = (b - MID) * scales[None, :] + zeros[None, :]

        acc += tl.dot(a, b)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    acc.to(tl.bfloat16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, sem='release')