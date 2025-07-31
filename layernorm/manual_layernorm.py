import torch

# Input tensor: shape (1, 4)
x = torch.tensor([[4.0, 6.0, 10.0, 0.0]])

# Step 1: Mean
mean = x.mean(dim=-1, keepdim=True)  # shape: (1, 1)

# Step 2: Std Dev
var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
std = torch.sqrt(var + 1e-5)

# Step 3: Normalize
x_norm = (x - mean) / std  # shape: (1, 4)

# Step 4: Apply learnable gamma (scale) and beta (shift)
# We'll define them manually here
gamma = torch.tensor([[1.0, 2.0, 1.0, 0.5]])  # shape: (1, 4)
beta = torch.tensor([[0.0, 1.0, 0.0, -1.0]])  # shape: (1, 4)

# Final LayerNorm output
# gamma and beta are initialized to 1 and 0 respectively,
# tracked as nn.Parameter(), and updated during backpropagation

output = gamma * x_norm + beta

print("Manual LayerNorm with Scale and Shift:", output)
