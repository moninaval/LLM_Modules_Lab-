import torch
import torch.nn.functional as F

# Step 1: Define input vector (1D)
x = torch.tensor([1.0, -2.0])  # shape: (2,)

# Step 2: Define W1 (2x4) and b1 (4)
W1 = torch.tensor([
    [1.0, -1.0, 0.0, 2.0],
    [0.0,  3.0, -1.0, 1.0]
])  # shape: (2, 4)

b1 = torch.tensor([0.0, 0.0, 0.0, 0.0])  # shape: (4,)

# Step 3: First linear transformation
x1 = x @ W1 + b1  # shape: (4,)
print("After first linear (x @ W1 + b1):", x1.tolist())

# Step 4: ReLU activation
x_relu = F.relu(x1)
print("After ReLU:", x_relu.tolist())

# Step 5: Define W2 (4x2) and b2 (2)
W2 = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [1.0, -1.0]
])  # shape: (4, 2)

b2 = torch.tensor([0.0, 0.0])  # shape: (2,)

# Step 6: Second linear transformation
x2 = x_relu @ W2 + b2  # shape: (2,)
print("Final FFN output (x2):", x2.tolist())
