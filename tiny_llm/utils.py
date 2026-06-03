import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch import device


# 1. Load Text
def read_text(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


# 2. Prepare Text
def prepare_text(raw_text: str) -> list[str]:
    END_TOKEN = "<END>"

    lines = [
        line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".").strip().lower()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    lines = [
        line for line in lines if not (line.startswith("=") and line.endswith("="))
    ]

    return [f"{line} {END_TOKEN}" for line in lines]


# 3. Create Tokens
def find_tokens(
    text: list[str],
    unknown_character: str = "<UNK>",
    filename: str = "tiny_llm/output_model/tokenizer.json",
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=unknown_character))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=20000,
        special_tokens=[
            unknown_character,
            "<|endoftext|>",
        ],
    )

    tokenizer.train_from_iterator(
        text,
        trainer,
    )

    tokenizer.save(filename)

    return tokenizer


# 4. Training sequences
def training_sequences(
    text: list[str], tokenizer, sequence_length: int, max_sequences: int
) -> list[tuple]:
    training_sequences = []

    for line in text:
        token_ids = tokenizer.encode(line).ids

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

    return training_sequences


# 5. Embedding tables
def create_embedding_tables(
    embedding_dimension: int, vocab_size: int, sequence_length: int, device: device
):
    embedding_table = nn.Parameter(
        torch.randn(vocab_size, embedding_dimension, device=device) * 0.01
    )

    position_embedding_table = nn.Parameter(
        torch.randn(sequence_length, embedding_dimension, device=device) * 0.01
    )

    return embedding_table, position_embedding_table


# 6. LM head tables
def create_lmhead_tables(embedding_dimension: int, vocab_size: int, device: device):
    return nn.Parameter(
        torch.randn(
            embedding_dimension,
            vocab_size,
            device=device,
        )
        * 0.01
    )


# 7. LayerNorm tables
def create_layernorm_tables(embedding_dimension: int, device: device):
    norm1_gamma = nn.Parameter(torch.ones(embedding_dimension, device=device))
    norm1_beta = nn.Parameter(torch.zeros(embedding_dimension, device=device))
    norm2_gamma = nn.Parameter(torch.ones(embedding_dimension, device=device))
    norm2_beta = nn.Parameter(torch.zeros(embedding_dimension, device=device))

    return norm1_gamma, norm1_beta, norm2_gamma, norm2_beta


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * ((x - mean) / (std + eps)) + beta


# 8. FeedForward tables
def create_feedforward_tables(embedding_dimension: int, device: device):
    feed_forward_dimension = embedding_dimension * 4

    W1 = nn.Parameter(
        torch.randn(embedding_dimension, feed_forward_dimension, device=device) * 0.01
    )
    b1 = nn.Parameter(torch.zeros(feed_forward_dimension, device=device))

    W2 = nn.Parameter(
        torch.randn(feed_forward_dimension, embedding_dimension, device=device) * 0.01
    )
    b2 = nn.Parameter(torch.zeros(embedding_dimension, device=device))

    return W1, b1, W2, b2


# 9. Attention tables
def create_attention_tables(
    embedding_dimension: int, attention_dimension: int, device: device
):
    W_Q = nn.Parameter(
        torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
    )
    W_K = nn.Parameter(
        torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
    )
    W_V = nn.Parameter(
        torch.randn(embedding_dimension, attention_dimension, device=device) * 0.01
    )
    W_O = nn.Parameter(
        torch.randn(attention_dimension, embedding_dimension, device=device) * 0.01
    )

    return W_Q, W_K, W_V, W_O


# 10. Optimizer
def create_optimizer(parameters: list, learning_rate: float):
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
    )


def get_batch(training_sequences: list[tuple], step: int, batch_size: int):
    return training_sequences[step : step + batch_size]


def get_batch_inputs(batch: list[tuple], device: device):
    return torch.tensor(
        [x[0] for x in batch],
        device=device,
    )


def get_batch_targets(batch: list[tuple], device: device):
    return torch.tensor(
        [x[1] for x in batch],
        device=device,
    )
