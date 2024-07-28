import torch

def uniform_quantize(x, bits=4):
    qmin, qmax = 0, 2**bits - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = qmin - torch.round(x.min() / scale)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return scale * (q - zero_point)