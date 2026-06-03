import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tiny_llm.utils import (
    create_attention_tables,
    create_embedding_tables,
    create_feedforward_tables,
    create_layernorm_tables,
    create_lmhead_tables,
    create_optimizer,
    find_tokens,
    get_batch,
    get_batch_inputs,
    get_batch_targets,
    layer_norm,
    prepare_text,
    read_text,
    training_sequences,
)

# 0. GPU Set up
### Assuming using mac
device = torch.device("mps")
print("Using Apple GPU:", device)

# 1. Load text
raw_text = read_text(filename="./input_data/TinyStories-valid.txt")

# 2 Prepare text
text = prepare_text(raw_text=raw_text)

# 3. Create Tokens
tokens = find_tokens(text=text)

# 4. Training sequences
sequence_length = 16

training_sequences = training_sequences(
    text=text,
    tokenizer=tokens,
    sequence_length=sequence_length,
    max_sequences=500_000,
)

# 5. Embedding tables
embedding_dimension = 64

embedding_table, position_embedding_table = create_embedding_tables(
    embedding_dimension,
    tokens.get_vocab_size(),
    len(training_sequences),
    device,
)

# 6. LM head tables
lm_head = create_lmhead_tables(
    embedding_dimension,
    tokens.get_vocab_size(),
    device,
)

# 7. LayerNorm tables
norm1_gamma, norm1_beta, norm2_gamma, norm2_beta = create_layernorm_tables(
    embedding_dimension,
    device,
)

# 8. FeedForward tables
W1, b1, W2, b2 = create_feedforward_tables(
    embedding_dimension,
    device,
)

# 9. Attention tables
attention_dimension = 64

W_Q, W_K, W_V, W_O = create_attention_tables(
    embedding_dimension,
    attention_dimension,
    device,
)

# 10. Create Optimizer
optimizer = create_optimizer(
    parameters=[
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
    ],
    learning_rate=0.001,
)

n_heads = 4
head_dimension = attention_dimension // n_heads
epochs = 20
batch_size = 64

# 11. Training Loop
for epoch in range(epochs):
    total_loss = 0
    random.shuffle(training_sequences)

    for step in tqdm(
        range(0, len(training_sequences), batch_size),
        desc=f"Epoch {epoch + 1}/{epochs}",
        unit="batch",
    ):
        # 11.1: batch of input ids and target id
        batch = get_batch(
            training_sequences=training_sequences,
            step=step,
            batch_size=batch_size,
        )

        # 11.2: get input ids
        batch_input_ids = get_batch_inputs(
            batch=batch,
            device=device,
        )

        # 11.3 get target ids
        batch_target_ids = get_batch_targets(
            batch=batch,
            device=device,
        )

        # 11.4 get embeddings "What word is this?"
        X = embedding_table[batch_input_ids]

        # 11.5 get position embeddings "Where is this word?"
        Y = torch.arange(sequence_length, device=device).unsqueeze(0)

        # 11.6 add them
        Z = X + position_embedding_table[Y]

        # 11.7 normalize
        Z_norm = layer_norm(
            Z,
            norm1_gamma,
            norm1_beta,
        )

        # 11.8 self-attention
        Q = Z_norm @ W_Q
        K = Z_norm @ W_K
        V = Z_norm @ W_V

        # 11.9 reshape
        current_batch_size = batch_input_ids.shape[0]

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

        # 11.10 attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / torch.sqrt(
            torch.tensor(
                head_dimension,
                device=device,
            )
        )

        # 11.11 causal masking
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
            ),
            diagonal=1,
        )

        # 11.12 scores
        scores = scores.masked_fill(
            mask == 1,
            float("-inf"),
        )

        # 11.13 attention softmax
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
        Z = Z + attention_output

        # layernorm 2
        Z_norm = layer_norm(
            Z,
            norm2_gamma,
            norm2_beta,
        )

        # feed forward network
        hidden = Z_norm @ W1 + b1
        hidden = F.gelu(hidden)
        ffn_output = hidden @ W2 + b2

        # residual connection 2
        Z = Z + ffn_output

        # lm head -> logits
        last_token_vector = Z[:, -1]
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
