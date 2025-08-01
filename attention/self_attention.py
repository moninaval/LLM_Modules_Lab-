import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Xavier initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, X):
        Q = self.W_q(X)  # (seq_len, d_model)
        K = self.W_k(X)
        V = self.W_v(X)

        print("Input X:\n", X)
        print("Query Q:\n", Q)
        print("Key K:\n", K)
        print("Value V:\n", V)

        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        print("Dot Product Scores:\n", scores)

        attention_weights = F.softmax(scores, dim=-1)
        print("Attention Weights:\n", attention_weights)

        output = attention_weights @ V
        print("Self-Attention Output:\n", output)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)  # for reproducibility

    # 3 tokens, each of dimension 4
    X = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0, 0.0]])

    model = SelfAttention(d_model=4)
    output = model(X)
