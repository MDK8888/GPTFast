#pragma once

__global__ void int4_matmul_unpack32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ activation,
    const int32_t* __restrict__ packed_weights,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int out_width,
    int out_height,
    int weight_width,
    int weight_height,
    int groupsize
);