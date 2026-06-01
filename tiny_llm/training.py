import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# 1. LOAD WIKI TRAINING DATA
# =============================
output_file = "tiny_llm/output_model/tiny_wiki_gpt.npz"
input_file = "tiny_llm/input_data/wiki.train.raw"

with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

lines = [line.strip().lower() for line in raw_text.splitlines() if line.strip()]

print("lines: ", len(lines))
print(lines[:3])

# ============================
# 2. BUILD VOCAB
# ============================

vocab = sorted(set(word for line in lines for word in line.split()))
vocab_size = len(vocab)

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

print("vocab_size")

# ============================
# 3. CREATE TRAINING SEQUENCES
# ============================

sequence_length = 8
max_sequences = 50_000

training_sequences = []

for line in lines:
    token_ids = [word_to_id[word] for word in line.split()]

    if len(token_ids) <= sequence_length:
        continue

    for i in range(len(token_ids) - sequence_length):
        input_ids = token_ids[i : i + sequence_length]
        target_id = token_ids[i + sequence_length]

        training_sequences.append((input_ids, target_id))

        if len(training_sequences) >= max_sequences:
            break

    if len(training_sequences) >= max_sequences:
        break

print("training_sequences: ", len(training_sequences))

# ============================
# 4. Model Settings
# ============================

embedding_dimension = 64
attention_dimension = 64
learning_rate = 0.001
epochs = 5


# ============================
# 5. Trainable Weights
# ============================

embedding_table = nn.Parameter(torch.randn(vocab_size, embedding_dimension) * 0.01)

W_Q = nn.Parameter(torch.randn(embedding_dimension, attention_dimension) * 0.01)
W_K = nn.Parameter(torch.randn(embedding_dimension, attention_dimension) * 0.01)
W_V = nn.Parameter(torch.randn(embedding_dimension, attention_dimension) * 0.01)

lm_head = nn.Parameter(torch.randn(attention_dimension, vocab_size))

parameters = [embedding_table, W_Q, W_K, W_V, lm_head]

optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# =========================
# 6. TRAINING LOOP
# =========================

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(training_sequences)

    for step, (input_ids, target_id) in enumerate(training_sequences):
        input_ids = torch.tensor(input_ids)
        target_id = torch.tensor(target_id)

        # embedding lookup
        X = embedding_table[input_ids]

        # self-attention
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        scores = Q @ K.T
        scores = scores / torch.sqrt(torch.tensor(attention_dimension))

        # causal mask
        mask = torch.triu(
            torch.ones(scores.shape),
            diagonal=1,
        )

        scores = scores.masked_fill(mask == 1, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        context = attention_weights @ V

        # predict next word from last token

        last_token_vector = context[-1]

        logits = last_token_vector @ lm_head

        loss = F.cross_entropy(
            logits.unsqueeze(0),
            target_id.unsqueeze(0),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 5000 == 0:
            print("epoch: ", epoch, " step: ", step, " loss: ", loss.item())

    avg_loss = total_loss / len(training_sequences)
    print("EPOCH Done: ", epoch, " avg loss: ", avg_loss)

np.savez(
    output_file,
    embedding_table=embedding_table.detach().cpu().numpy(),
    W_Q=W_Q.detach().cpu().numpy(),
    W_K=W_K.detach().cpu().numpy(),
    W_V=W_V.detach().cpu().numpy(),
    lm_head=lm_head.detach().cpu().numpy(),
    vocab=np.array(vocab),
    sequence_length=sequence_length,
    embedding_dimension=embedding_dimension,
    attention_dimension=attention_dimension,
)

print(f"Saved model to {output_file}")
