import torch

def pack_int4_weights(weights:torch.Tensor):
    """
    Pack int4 weights horizontally.
    
    Args:
    weights (torch.Tensor): Input tensor of shape (In, Out) with values in range [0, 15]
    
    Returns:
    torch.Tensor: Packed weights of shape (In, Out // 8)
    """
    In, Out = weights.shape
    assert Out % 8 == 0, "Out dimension must be divisible by 8"
    
    # Reshape to (In, Out // 8, 8)
    reshaped = weights.reshape(In, Out // 8, 8)
    
    # Pack 8 int4 values into one int32
    packed = torch.zeros((In, Out // 8), dtype=torch.int32, device=weights.device)
    for i in range(8):
        packed |= reshaped[:, :, i].to(torch.int32) << (4 * i)
    
    return packed