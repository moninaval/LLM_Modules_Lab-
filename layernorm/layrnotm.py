import torch
import torch.nn as nn

x = torch.tensor([[4.0, 6.0, 10.0, 0.0]])

# Create LayerNorm with affine params enabled (default)
layer_norm = nn.LayerNorm(normalized_shape=4, eps=1e-5)

# Manually set gamma (weight) and beta (bias)
with torch.no_grad():
    layer_norm.weight.copy_(torch.tensor([1.0, 2.0, 1.0, 0.5]))
    layer_norm.bias.copy_(torch.tensor([0.0, 1.0, 0.0, -1.0]))

output_pytorch = layer_norm(x)

print("PyTorch LayerNorm with Scale and Shift:", output_pytorch)
