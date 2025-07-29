import torch
import torch.nn as nn

# 1. Setup
vocab_size = 10
embed_dim = 4

# 2. Create embedding layer (try with and without max_norm)
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, max_norm=1.0)

# 3. Print raw embedding matrix
print("🔹 Raw Embedding Weight Matrix:")
print(embedding.weight)
print("-" * 50)

# 4. Define example token IDs (e.g., a sentence)
token_ids = torch.tensor([1, 3, 5, 3])  # shape: [seq_len]

# 5. Get embedded vectors
embedded = embedding(token_ids)

# 6. Print result
print("🔹 Token IDs:", token_ids)
print("🔹 Embedded Vectors (after max_norm if enabled):")
print(embedded)

# 7. Show L2 norms of the selected vectors
norms = torch.norm(embedded, dim=1)
print("🔹 Norms of embedded vectors:", norms)
