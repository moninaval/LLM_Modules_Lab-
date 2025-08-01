import torch
import torch.nn.functional as F

# Step 1: Define a 3-token input sequence (each token = 4-dim vector)
X = torch.tensor([[1.0, 0.0, 1.0, 0.0],   # Token 0
                  [0.0, 1.0, 0.0, 1.0],   # Token 1
                  [1.0, 1.0, 0.0, 0.0]])  # Token 2

# Step 2: Manually define projection matrices for Q, K, V
W_q = torch.tensor([[1., 0., 0., 0.],    # Q = X @ W_q^T
                    [0., 1., 0., 0.]])
W_k = torch.tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.]])
W_v = torch.tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.]])

# Step 3: Project Q, K, V
Q = X @ W_q.T   # shape: (3, 2)
K = X @ W_k.T
V = X @ W_v.T

print("Token 0 Q vector:", Q[0])
print("All K vectors:\n", K)

# Step 4: Compute attention scores for token 0 (row 0 of attention matrix)
scores_token0 = Q[0] @ K.T  # shape: (3,)
print("\nAttention raw scores (Q[0] • K[j]):", scores_token0)

# Step 5: Apply softmax to get attention weights
attn_weights_token0 = F.softmax(scores_token0, dim=-1)
print("Attention weights for token 0:", attn_weights_token0)

# Step 6: Weighted sum of V using attention weights
attended_vector0 = attn_weights_token0 @ V
print("\nSelf-attention output for token 0:", attended_vector0)
