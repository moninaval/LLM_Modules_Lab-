import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: [batch, seq_len, d_k]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

if __name__ == "__main__":
    # Batch=1, seq_len=2, d_model=4, d_k=2
    batch_size = 1
    seq_len = 2
    d_model = 4
    d_k = d_v = 2

    # Set token embeddings manually (X)
    X = torch.tensor([[[1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0, 1.0]]])  # shape: [1, 2, 4]

    print("🔹 Input X:")
    print(X)

    # Define simple Wq, Wk, Wv with fixed weights
    Wq = torch.nn.Linear(d_model, d_k, bias=False)
    Wk = torch.nn.Linear(d_model, d_k, bias=False)
    Wv = torch.nn.Linear(d_model, d_v, bias=False)

    # Manually set Wq, Wk, Wv weights
    Wq.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0],    # picks dim 0 and 1
                                   [0.0, 1.0, 0.0, 0.0]])   # picks dim 1
    Wk.weight.data = torch.tensor([[0.0, 0.0, 1.0, 0.0],    # picks dim 2
                                   [0.0, 0.0, 0.0, 1.0]])   # picks dim 3
    Wv.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0],    # picks dim 0
                                   [0.0, 1.0, 0.0, 0.0]])   # picks dim 1

    # Compute Q, K, V
    Q = Wq(X)  # shape: [1, 2, 2]
    K = Wk(X)
    V = Wv(X)

    print("\n🔸 Q:")
    print(Q)
    print("\n🔸 K:")
    print(K)
    print("\n🔸 V:")
    print(V)

    # Apply scaled dot-product attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print("\n✅ Attention Weights:")
    print(attn_weights)

    print("\n✅ Attention Output:")
    print(output)
