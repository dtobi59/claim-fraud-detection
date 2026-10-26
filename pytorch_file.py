import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Hyperparameters
vocab_size = 6        # Total unique tokens in vocabulary
d_model = 4           # Model dimension (length of vector per token)
seq_len = 3           # Number of tokens in input sequence

# 2. Vocabulary Mapping
# {"the": 0, "king": 1, "ruled": 2, "queen": 3, "palace": 4, "<unk>": 5}
input_tokens = torch.tensor([0, 1, 2])  # "the king ruled"

# 3. Embedding Layer (Lookup Table: vocab_size x d_model)
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

# Grab the d_model vectors for our 3 tokens
# Shape: (seq_len, d_model) -> (3, 4)
X = embedding(input_tokens)
print("Token Vectors (Shape 3x4):\n", X.detach())

# -------------------------------------------------------------
# 4. Scaled Dot-Product Attention: softmax(Q @ K.T / sqrt(d_k)) @ V
# -------------------------------------------------------------

# Linear projections for Q, K, V
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(X)  # (3, 4)
K = W_k(X)  # (3, 4)
V = W_v(X)  # (3, 4)

# Raw Attention Scores: Q @ K^T -> (3, 3)
scores = torch.matmul(Q, K.transpose(-2, -1))

# Scale by sqrt(d_k)
d_k = d_model
scaled_scores = scores / (d_k ** 0.5)

# Softmax over rows to get probabilities (each row sums to 1)
attention_weights = F.softmax(scaled_scores, dim=-1)
print("\nAttention Matrix (Shape 3x3):\n", attention_weights.detach())

# Weighted sum of values: Weights @ V -> (3, 4)
output = torch.matmul(attention_weights, V)
print("\nFinal Attention Output (Shape 3x4):\n", output.detach())