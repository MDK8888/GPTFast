// File: int4_matmul_wrapper.cpp

#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Include the headers for your CUDA kernels
#include "GPTQ/packing/int4_pack.cuh"
#include "GPTQ/matmul/int4_matmul.cuh"

// Declare the CUDA kernel
extern "C" __global__ void int4_matmul_unpack32_kernel(
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

__global__ void int4_pack32_kernel(
    int32_t* __restrict__ packed_weights,
    const int32_t* __restrict__ quantized_weights,
    int in_features,
    int out_features
);

// C++ wrapper function
torch::Tensor int4_matmul_unpack32_cuda(
    torch::Tensor activation,
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zeros,
    int groupsize
) {
    // Get tensor dimensions
    auto activation_sizes = activation.sizes();
    int64_t batch_size = 1;
    for (int i = 0; i < activation_sizes.size() - 1; ++i) {
        batch_size *= activation_sizes[i];
    }
    int64_t in_features = activation_sizes.back();
    int64_t out_features = packed_weights.size(1) * 8;  // Unpack the output dimension

    // Reshape activation if necessary
    activation = activation.reshape({batch_size, in_features});

    // Ensure all inputs are on CUDA
    activation = activation.to(torch::kCUDA).to(torch::kBFloat16);
    packed_weights = packed_weights.to(torch::kCUDA).to(torch::kInt32);
    scales = scales.to(torch::kCUDA).to(torch::kFloat32);
    zeros = zeros.to(torch::kCUDA).to(torch::kFloat32);

    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto output = torch::empty({batch_size, out_features}, options);

    // Set up grid and block dimensions
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 8;
    dim3 gridDim((out_features + WMMA_N - 1) / WMMA_N, (batch_size + WMMA_M - 1) / WMMA_M);
    dim3 blockDim(1, 1);

    // Launch kernel
    int4_matmul_unpack32_kernel<<<gridDim, blockDim>>>(
        output.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr<at::BFloat16>()),
        packed_weights.data_ptr<int32_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        out_features,
        batch_size,
        packed_weights.size(1),
        in_features,
        groupsize
    );

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    return output;
}

torch::Tensor pack_int4_weights(torch::Tensor quantized_weights) {
    TORCH_CHECK(quantized_weights.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(quantized_weights.dtype() == torch::kInt32, "Input must be int32");
    TORCH_CHECK(quantized_weights.is_cuda(), "Input must be a CUDA tensor");
    
    auto in_features = quantized_weights.size(0);
    auto out_features = quantized_weights.size(1);
    
    TORCH_CHECK(out_features % 8 == 0, "out_features must be divisible by 8");

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(quantized_weights.device());
    auto packed_weights = torch::empty({in_features, out_features / 8}, options);

    // Launch kernel
    dim3 block_size(32, 32);
    dim3 grid_size((out_features / 8 + block_size.x - 1) / block_size.x, 
                   (in_features + block_size.y - 1) / block_size.y);

    int4_pack32_kernel<<<grid_size, block_size>>>(
        packed_weights.data_ptr<int32_t>(),
        quantized_weights.data_ptr<int32_t>(),
        in_features,
        out_features
    );

    return packed_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_matmul_unpack32", &int4_matmul_unpack32_cuda, "INT4 MatMul Unpack32 (CUDA)");
    m.def("pack_int4_weights", &pack_int4_weights, "Pack int4 weights into int32");
}