import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tiny_llm.utils import find_tokens, prepare_text, read_text, training_sequences

# 0. GPU Set up
### Assuming using mac
device = torch.device("mps")
print("Using Apple GPU:", device)

# 1. Load text
raw_text = read_text(filename="./input_data/TinyStories-valid.txt")

# 2 Prepare text
text = prepare_text(raw_text=raw_text)

# 3. Find Tokens
tokens = find_tokens(text=text)

# 4. Build training sequences
training_sequences = training_sequences(
    text=text,
    tokenizer=tokens,
    sequence_length=16,
    max_sequences=500_000,
)

# 5. Weights & Tables
embedding_dimension = 64
attention_dimension = 64
feed_forward_dimension = embedding_dimension * 4

learning_rate = 0.001
epochs = 20
batch_size = 64

n_heads = 4
head_dimension = attention_dimension // n_heads


## tables
embedding_table = nn.Parameter(
    torch.randn(vocab_size, embedding_dimension, device=device) * 0.01
)

position_embedding_table = nn.Parameter(
    torch.randn(sequence_length, embedding_dimension, device=device) * 0.01
)

## Attention Weights

W_Q = nn.Parameter(
    torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
)
W_K = nn.Parameter(
    torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
)
W_V = nn.Parameter(
    torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
)

# Output projection after attention
W_O = nn.Parameter(
    torch.randn(attention_dimension, embedding_dimension, device=device) * 0.01
)

# LayerNorm weights
norm1_gamma = nn.Parameter(torch.ones(embedding_dimension, device=device))
norm1_beta = nn.Parameter(torch.zeros(embedding_dimension, device=device))

norm2_gamma = nn.Parameter(torch.ones(embedding_dimension, device=device))
norm2_beta = nn.Parameter(torch.zeros(embedding_dimension, device=device))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * ((x - mean) / (std + eps)) + beta


# Feed Forward Network weights
W1 = nn.Parameter(
    torch.randn(embedding_dimension, feed_forward_dimension, device=device) * 0.01
)
b1 = nn.Parameter(torch.zeros(feed_forward_dimension, device=device))

W2 = nn.Parameter(
    torch.randn(feed_forward_dimension, embedding_dimension, device=device) * 0.01
)
b2 = nn.Parameter(torch.zeros(embedding_dimension, device=device))

# LM head
lm_head = nn.Parameter(
    torch.randn(embedding_dimension, vocab_size, device=device) * 0.01
)

parameters = [
    embedding_table,
    position_embedding_table,
    W_Q,
    W_K,
    W_V,
    W_O,
    norm1_gamma,
    norm1_beta,
    norm2_gamma,
    norm2_beta,
    W1,
    b1,
    W2,
    b2,
    lm_head,
]

optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# 5. Training Loop
for epoch in range(epochs):
    total_loss = 0
    random.shuffle(training_sequences)

    for step in tqdm(
        range(0, len(training_sequences), batch_size),
        desc=f"Epoch {epoch + 1}/{epochs}",
        unit="batch",
    ):
        # input ids -> target ids
        batch = training_sequences[step : step + batch_size]

        batch_input_ids = torch.tensor(
            [x[0] for x in batch],
            device=device,
        )

        batch_target_ids = torch.tensor(
            [x[1] for x in batch],
            device=device,
        )

        current_batch_size = batch_input_ids.shape[0]

        # Lookup input_id's embeddings
        X = embedding_table[batch_input_ids]

        # Lookup position embeddings
        positions = torch.arange(sequence_length, device=device)
        positions = positions.unsqueeze(0)

        X = X + position_embedding_table[positions]

        # normalization | Layernorm 1
        X_norm = layer_norm(
            X,
            norm1_gamma,
            norm1_beta,
        )

        # self-attention
        Q = X_norm @ W_Q
        K = X_norm @ W_K
        V = X_norm @ W_V

        Q = Q.reshape(
            current_batch_size,
            sequence_length,
            n_heads,
            head_dimension,
        ).transpose(1, 2)

        K = K.reshape(
            current_batch_size,
            sequence_length,
            n_heads,
            head_dimension,
        ).transpose(1, 2)

        V = V.reshape(
            current_batch_size,
            sequence_length,
            n_heads,
            head_dimension,
        ).transpose(1, 2)

        # attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(
            torch.tensor(
                head_dimension,
                device=device,
            )
        )

        # causal masking
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask == 1,
            float("-inf"),
        )

        # attention softmax
        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        # context mixing
        attention_output = attention_weights @ V

        attention_output = attention_output.transpose(1, 2).reshape(
            current_batch_size,
            sequence_length,
            attention_dimension,
        )

        # output projection
        attention_output = attention_output @ W_O

        # residual connection 1
        X = X + attention_output

        # layernorm 2
        X_norm = layer_norm(
            X,
            norm2_gamma,
            norm2_beta,
        )

        # feed forward network
        hidden = X_norm @ W1 + b1
        hidden = F.gelu(hidden)
        ffn_output = hidden @ W2 + b2

        # residual connection 2
        X = X + ffn_output

        # lm head -> logits
        last_token_vector = X[:, -1]
        logits = last_token_vector @ lm_head

        # loss

        loss = F.cross_entropy(
            logits,
            batch_target_ids,
        )

        # backprop + weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(training_sequences) / batch_size)
    print(f"\nEPOCH {epoch + 1}/{epochs} DONE | avg loss = {avg_loss:.4f}")
