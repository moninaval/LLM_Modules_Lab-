import torch
import torch.nn.functional as F

# --- Hyperparameters ---
batch_size = 1
seq_len = 5
d_model = 16
num_heads = 4
d_k = d_model // num_heads

# --- Fake Input ---
X = torch.randn(batch_size, seq_len, d_model)  # (1, 5, 16)

# --- Create Q, K, V weight matrices for all heads ---
W_Q = torch.randn(num_heads, d_model, d_k)     # (4, 16, 4)
W_K = torch.randn(num_heads, d_model, d_k)     # (4, 16, 4)
W_V = torch.randn(num_heads, d_model, d_k)     # (4, 16, 4)
W_O = torch.randn(num_heads * d_k, d_model)    # (16, 16)

# --- Store outputs from all heads ---
all_heads = []

for i in range(num_heads):
    # Project Q, K, V using that head's weights
    Q = X @ W_Q[i]   # (1, 5, 4)
    K = X @ W_K[i]   # (1, 5, 4)
    V = X @ W_V[i]   # (1, 5, 4)

    # Scaled dot-product attention
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (1, 5, 5)
    attn_weights = F.softmax(scores, dim=-1)         # (1, 5, 5)
    head_output = attn_weights @ V                  # (1, 5, 4)

    all_heads.append(head_output)

# --- Concatenate heads ---
concat_heads = torch.cat(all_heads, dim=-1)          # (1, 5, 16)

# --- Final linear projection ---
output = concat_heads @ W_O                          # (1, 5, 16)

# --- Final output ---
print("Final Multi-Head Attention Output Shape:", output.shape)
print("Output Tensor:\n", output)
