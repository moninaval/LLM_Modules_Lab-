import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, X):
        seq_len = X.size(0)

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Generate causal mask (upper triangular is masked)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        print("Causal Mask:\n", causal_mask)

        # Apply mask: replace future scores with -inf
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V

        print("Q:\n", Q)
        print("K:\n", K)
        print("V:\n", V)
        print("Masked Attention Scores:\n", scores)
        print("Attention Weights:\n", attention_weights)
        print("Causal Self-Attention Output:\n", output)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    # 4 tokens, each of dim 4
    X = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0, 0.0],
                      [0.5, 0.5, 0.5, 0.5]])

    model = CausalSelfAttention(d_model=4)
    output = model(X)
