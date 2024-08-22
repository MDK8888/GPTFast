import torch
import torch.nn as nn
from GPTFast.Core.Quantize import GPTQLinearModuleQuantizer, WeightOnlyInt4Linear
from GPTFast.Kernels import int4_matmul, pack_int4_weights
from transformers.pytorch_utils import Conv1D

def test_int4_matmul_kernel_correctness():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Create a random linear layer with adjusted dimensions
    in_features, out_features = 128, 256  # Adjusted dimensions
    linear_layer = Conv1D(out_features, in_features) #nn.Linear(in_features, out_features, bias=False, dtype=torch.float16, device="cuda")
    linear_layer = linear_layer.to(torch.float16).to("cuda")

    # Create random input data
    batch_size = 1  # Adjusted batch size
    input_data = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")

    # Initialize the GPTQLinearModuleQuantizer
    quantizer = GPTQLinearModuleQuantizer(linear_layer, "deez_nuts", blocksize=128, percdamp=0.01, groupsize=32, dtype=torch.float16, device="cuda")

    # Add batches of data
    num_batches = 10
    for _ in range(num_batches):
        with torch.no_grad():
            output = linear_layer(input_data)
        quantizer.add_batch(input_data, output)

    # Perform quantization
    Q, DQ, qparams = quantizer.quantize()

    # Pack the quantized weights
    packed_weights = pack_int4_weights(Q).t().contiguous()

    # Extract scales and zeros from qparams
    scales, zeros = qparams
    scales = scales.to("cuda")
    zeros = zeros.to("cuda")

    print("Packed weights shape:", packed_weights.shape)
    print("Scales shape:", scales.shape)
    print("Zeros shape:", zeros.shape)

    weight_only_int4_linear = WeightOnlyInt4Linear(in_features, out_features, groupsize=32)
    weight_only_int4_linear.weight = packed_weights.to("cuda")
    weight_only_int4_linear.scales = scales.to("cuda")
    weight_only_int4_linear.zeros = zeros.to("cuda")
    # Create nn.Linear with DQ (without bias)
    linear_dq = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).to("cuda")
    linear_dq.weight.data = DQ

    # Test with input data
    test_input = torch.randn(batch_size, in_features, dtype=torch.float16).to("cuda")

    # Forward pass through nn.Linear with DQ
    with torch.no_grad():
        dq_output = linear_dq(test_input)

    # Run our int4_matmul kernel
    int4_matmul_kernel_output = weight_only_int4_linear(test_input)

    dq_output = dq_output.to(torch.float16)

    # Compare outputs
    mse_loss = nn.MSELoss()
    int4_matmul_kernel_dq_error = mse_loss(int4_matmul_kernel_output, dq_output)
    print(f"\nError between int4_matmul kernel and nn.Linear with DQ (MSE): {int4_matmul_kernel_dq_error.item()}")

    # Calculate relative error
    relative_error = torch.abs(int4_matmul_kernel_output - dq_output) / (torch.abs(dq_output) + 1e-8)  # Added small epsilon to avoid division by zero
    max_relative_error = torch.max(relative_error)
    print(f"\nMaximum relative error between int4_matmul kernel and nn.Linear with DQ: {max_relative_error.item()}")

    # Calculate the percentage of elements with relative error less than 1e-3
    percentage_small_error = (relative_error < 1e-3).float().mean() * 100
    print(f"Percentage of elements in int4_matmul_kernel_output with relative error < 1e-3: {percentage_small_error.item():.2f}%")

    # Find and print indices and values of largest discrepancies
    num_largest = 10
    largest_rel_diff_indices = torch.topk(relative_error.view(-1), num_largest).indices

    print(f"\nTop {num_largest} Largest Relative Differences between int4_matmul_kernel_output and dq_output:")
    for idx in largest_rel_diff_indices:
        row, col = idx // dq_output.shape[1], idx % dq_output.shape[1]
        print(f"Index: ({row}, {col}), DQ: {dq_output[row, col].item():.6f}, "
              f"Kernel: {int4_matmul_kernel_output[row, col].item():.6f}, "
              f"Rel Diff: {relative_error[row, col].item():.6f}")

    # Check if the outputs are close enough
    tolerance = 1e-3
    is_close = torch.allclose(int4_matmul_kernel_output, dq_output, rtol=tolerance, atol=tolerance)
    print(f"\nOutputs are close within tolerance {tolerance}: {is_close}")

    # Print a sample of the output tensors
    print("\nSample of DQ output (first 5 rows, first 10 columns):")
    print(dq_output[:5, :10])
    print("\nSample of Kernel output (first 5 rows, first 10 columns):")
    print(int4_matmul_kernel_output[:5, :10])

if __name__ == "__main__":
    test_int4_matmul_kernel_correctness()