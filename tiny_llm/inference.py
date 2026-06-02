import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# =========================
# LOAD MODEL + TOKENIZER
# =========================

model_path = "tiny_llm/output_model/tiny_stories_gpt.npz"
tokenizer_path = "tiny_llm/output_model/tokenizer.json"

model = np.load(model_path, allow_pickle=True)

tokenizer = Tokenizer.from_file(tokenizer_path)

# =========================
# LOAD TRAINED WEIGHTS
# =========================

embedding_table = torch.tensor(model["embedding_table"], dtype=torch.float32)
position_embedding_table = torch.tensor(
    model["position_embedding_table"], dtype=torch.float32
)

W_Q = torch.tensor(model["W_Q"], dtype=torch.float32)
W_K = torch.tensor(model["W_K"], dtype=torch.float32)
W_V = torch.tensor(model["W_V"], dtype=torch.float32)
W_O = torch.tensor(model["W_O"], dtype=torch.float32)

norm1_gamma = torch.tensor(model["norm1_gamma"], dtype=torch.float32)
norm1_beta = torch.tensor(model["norm1_beta"], dtype=torch.float32)
norm2_gamma = torch.tensor(model["norm2_gamma"], dtype=torch.float32)
norm2_beta = torch.tensor(model["norm2_beta"], dtype=torch.float32)

W1 = torch.tensor(model["W1"], dtype=torch.float32)
b1 = torch.tensor(model["b1"], dtype=torch.float32)
W2 = torch.tensor(model["W2"], dtype=torch.float32)
b2 = torch.tensor(model["b2"], dtype=torch.float32)

lm_head = torch.tensor(model["lm_head"], dtype=torch.float32)

# =========================
# LOAD MODEL SETTINGS
# =========================

sequence_length = int(model["sequence_length"])
attention_dimension = int(model["attention_dimension"])
n_heads = int(model["n_heads"])
head_dimension = int(model["head_dimension"])

END_TOKEN = "<|endoftext|>"

# =========================
# LAYER NORM
# =========================


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * ((x - mean) / (std + eps)) + beta


# =========================
# AUTOREGRESSIVE GENERATION
# =========================


def generate_text(text, max_new_tokens=50, temperature=0.8):
    generated_ids = tokenizer.encode(text.lower()).ids

    for _ in range(max_new_tokens):
        input_ids = generated_ids[-sequence_length:]

        if len(input_ids) == 0:
            return text

        input_ids = torch.tensor(input_ids)

        with torch.no_grad():
            # embedding lookup
            X = embedding_table[input_ids]

            # position embeddings
            positions = torch.arange(len(input_ids))
            X = X + position_embedding_table[positions]

            # layernorm 1
            X_norm = layer_norm(X, norm1_gamma, norm1_beta)

            # multi-head self-attention
            Q = X_norm @ W_Q
            K = X_norm @ W_K
            V = X_norm @ W_V

            input_length = len(input_ids)

            Q = Q.reshape(input_length, n_heads, head_dimension).transpose(0, 1)
            K = K.reshape(input_length, n_heads, head_dimension).transpose(0, 1)
            V = V.reshape(input_length, n_heads, head_dimension).transpose(0, 1)

            # attention scores
            scores = Q @ K.transpose(-2, -1)
            scores = scores / torch.sqrt(torch.tensor(head_dimension))

            # causal masking
            mask = torch.triu(
                torch.ones(input_length, input_length),
                diagonal=1,
            )

            scores = scores.masked_fill(mask == 1, float("-inf"))

            # attention softmax
            attention_weights = F.softmax(scores, dim=-1)

            # context mixing
            attention_output = attention_weights @ V

            attention_output = attention_output.transpose(0, 1).reshape(
                input_length,
                attention_dimension,
            )

            # output projection
            attention_output = attention_output @ W_O

            # residual connection 1
            X = X + attention_output

            # layernorm 2
            X_norm = layer_norm(X, norm2_gamma, norm2_beta)

            # feed forward network
            hidden = X_norm @ W1 + b1
            hidden = F.gelu(hidden)
            ffn_output = hidden @ W2 + b2

            # residual connection 2
            X = X + ffn_output

            # lm head → logits
            last_token_vector = X[-1]
            logits = last_token_vector @ lm_head

            # temperature sampling
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)

            next_id = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

            next_token = tokenizer.id_to_token(next_id)

            if next_token == END_TOKEN:
                break

            generated_ids.append(next_id)

    return tokenizer.decode(generated_ids)


print(generate_text("once upon a time", max_new_tokens=25))
print(generate_text("the little dog", max_new_tokens=25))
print(generate_text("a girl went to", max_new_tokens=25))
