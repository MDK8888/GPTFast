__global__ void int4_pack32_kernel(
    int32_t* __restrict__ packed_weights,
    const int32_t* __restrict__ quantized_weights,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= in_features || col >= out_features / 8) {
        return;
    }

    int32_t packed_value = 0;
    for (int i = 0; i < 8; i++) {
        int original_col = col * 8 + i;
        if (original_col < out_features) {
            int32_t quant_value = quantized_weights[row * out_features + original_col];
            packed_value |= (quant_value & 0xF) << (i * 4);
        }
    }

    packed_weights[row * (out_features / 8) + col] = packed_value;
}

