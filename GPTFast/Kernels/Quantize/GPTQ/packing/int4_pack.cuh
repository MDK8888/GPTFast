#pragma once

__global__ void int4_pack32_kernel(
    int32_t* __restrict__ packed_weights,
    const int32_t* __restrict__ quantized_weights,
    int in_features,
    int out_features
);