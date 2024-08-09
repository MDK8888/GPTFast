#include <cuda_fp16.h>
#include <mma.h>

__device__ void int4_matmul_kernel(
    float* __restrict__ out,
    float* __restrict__ activation,
    float* __restrict__ quantized_weight,
    float* __restrict__ scales,
    float* __restrict__ zeros,
    int out_width,
    int out_height,
    int weight_width,
    int weight_height,
    int groupsize,
    int qmin,
    int qmax
){
    int matmul_dim = weight_height;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= out_height || col >= out_width) {
        return;
    }

    int mid = (int)((qmax + qmin + 1)/2);

    int sz_index = (int)(col / groupsize);
    int sz_width = (int)(weight_width / groupsize);

    for(int i = 0; i < matmul_dim; i++){
        float activation_element = activation[row * matmul_dim + i];
        float weight_element = quantized_weight[i * weight_width + col];
        weight_element-= mid;
        
        float scale_element = scales[i * sz_width + sz_index];
        weight_element *= scale_element;
        float zero_element = zeros[i * sz_width + sz_index];
        weight_element += zero_element; 

        out[row * out_width + col] += weight_element * activation_element;
    }
}

using namespace nvcuda;

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
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 32;
    constexpr int MID = 8;  // (qmin + qmax + 1) // 2, where qmin = 0, qmax = 15

    int row = blockIdx.y * WMMA_M;
    int col = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_act;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int4_t, wmma::col_major> frag_weight;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);

    // Pre-determine the column in the weight matrix
    int weight_col = blockIdx.x;

    for (int k = 0; k < weight_height; k += WMMA_K) {
        // Load activation fragment (16 x 32)
        if (row < out_height) {
            wmma::load_matrix_sync(frag_act, activation + row * weight_height + k, weight_height);
        } else {
            wmma::fill_fragment(frag_act, __float2bfloat16(0.0f));
        }

        // Load and unpack weight fragment (32 x 8)
        if (k < weight_height && col < weight_width) {
            for (int i = 0; i < WMMA_K; i++) {
                // Corrected: Use k as the starting row for our 32x1 weight fragment
                int packed_idx = (i + k) * weight_width + weight_col;
                int32_t packed_val = packed_weights[packed_idx];
                
                for (int j = 0; j < WMMA_N; j++) {
                    int fragment_idx = i * WMMA_N + j;
                    if (fragment_idx < WMMA_K * WMMA_N) {
                        frag_weight.x[fragment_idx] = (packed_val >> (j * 4)) & 0xF;
                    } else {
                        frag_weight.x[fragment_idx] = 0;
                    }
                }
            }
        } else {
            wmma::fill_fragment(frag_weight, 0);
        }

        // Perform matrix multiplication (16 x 32 * 32 x 8 = 16 x 8)
        wmma::mma_sync(frag_acc, frag_act, frag_weight, frag_acc);

        for (int i = 0; i < frag_acc.num_elements; i++) {
            int out_row = row + i / WMMA_N;
            int out_col = col + i % WMMA_N;
            if (out_row < out_height && out_col < out_width) {
                int out_idx = out_row * out_width + out_col;
                int group_idx = out_col / groupsize;
                float scale = scales[out_row * (weight_width / groupsize) + group_idx];
                float zero = zeros[out_row * (weight_width / groupsize) + group_idx];
                
                // Dequantize here, accounting for the quantization scheme
                float dequantized = (frag_acc.x[i] - MID) * scale + zero;
                out[out_idx] += dequantized; //add here, don't just set. In tiling, we need to add because each 16 x 8 product is only a piece of the total.
            }
        }
    }
    // Dequantize and store results
}