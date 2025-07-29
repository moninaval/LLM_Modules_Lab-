import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: tensors of shape [batch_size, seq_len, d_k]
    Returns:
        - attention output: [batch_size, seq_len, d_k]
        - attention weights: [batch_size, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    print("d_k size")
    print(torch.sqrt(torch.tensor(d_k, dtype=torch.float32)))

    # 1. Compute raw attention scores (QKᵀ / sqrt(d_k))
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    print("score")
    print(scores.shape)
    print(scores)

    # 2. Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax over last dim
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Weighted sum of V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# ----------------------------
# ✅ Test the function
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_k = 8

    # Create random Q, K, V
    Q = torch.rand(batch_size, seq_len, d_k)
    K = torch.rand(batch_size, seq_len, d_k)
    V = torch.rand(batch_size, seq_len, d_k)

    print("🔹 Q matrix (input queries):")
    print(Q)
    print("-" * 60)

    print("🔹 K matrix (input keys):")
    print(K)
    print("-" * 60)

    print("🔹 V matrix (input values):")
    print(V)
    print("-" * 60)

    # Compute scaled dot-product attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print("\n✅ Attention Output (QKᵀ / √d * V):")
    print(output)

    print("\n✅ Attention Weights (softmax scores):")
    print(attn_weights)

    # Sanity check on shapes
    assert output.shape == (batch_size, seq_len, d_k), "Output shape is incorrect"
    assert attn_weights.shape == (batch_size, seq_len, seq_len), "Attention weight shape is incorrect"

    print("\n✅ All tests passed!")
