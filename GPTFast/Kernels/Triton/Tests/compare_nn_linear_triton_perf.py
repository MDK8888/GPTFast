import torch
import torch.nn as nn
import time
from GPTFast.Kernels import int4_matmul, pack_int4_weights

def profile_and_compare():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Set up dimensions
    batch_size = 1
    in_features = 768
    out_features = 3072

    # Create input tensor
    input_tensor = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)

    # Create quantized weight tensor (simulating the output of quantization)
    weight = torch.randint(0, 16, (out_features, in_features), device='cuda', dtype=torch.float32)
    packed_weight = pack_int4_weights(weight).t().contiguous()

    # Create scales and zeros
    groupsize = 32
    num_groups = in_features // groupsize
    scales = torch.randn(out_features, num_groups, device='cuda', dtype=torch.float32)
    zeros = torch.randn(out_features, num_groups, device='cuda', dtype=torch.float32)

    # Create nn.Linear layer
    linear_layer = nn.Linear(in_features, out_features, bias=False).to(torch.float16).to('cuda')
    linear_layer_compiled = torch.compile(linear_layer)
    int4_matmul_compiled = torch.compile(int4_matmul)

    # Warm-up runs (10 iterations each)
    for _ in range(10):
        _ = int4_matmul_compiled(input_tensor, packed_weight, scales, zeros, groupsize)
        _ = linear_layer_compiled(input_tensor)

    # Number of iterations for timing
    num_iterations = 100

    # CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Profile int4_matmul
    start_event.record()
    for _ in range(num_iterations):
        output_int4 = int4_matmul_compiled(input_tensor, packed_weight, scales, zeros, groupsize)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_int4 = start_event.elapsed_time(end_event)
    avg_time_int4 = elapsed_time_int4 / num_iterations

    # Profile nn.Linear
    start_event.record()
    for _ in range(num_iterations):
        output_linear = linear_layer_compiled(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_linear = start_event.elapsed_time(end_event)
    avg_time_linear = elapsed_time_linear / num_iterations

    # Print results
    print(f"Average execution time for int4_matmul over {num_iterations} iterations: {avg_time_int4:.4f} ms")
    print(f"Average execution time for nn.Linear over {num_iterations} iterations: {avg_time_linear:.4f} ms")
    
    speedup = avg_time_linear / avg_time_int4
    print(f"Speedup of int4_matmul over nn.Linear: {speedup:.2f}x")

    # Ensure outputs are used to prevent potential optimizations
    print(f"int4_matmul output shape: {output_int4.shape}, sum: {output_int4.sum().item()}")
    print(f"nn.Linear output shape: {output_linear.shape}, sum: {output_linear.sum().item()}")

if __name__ == "__main__":
    profile_and_compare()