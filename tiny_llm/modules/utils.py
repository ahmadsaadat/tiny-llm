import torch
import torch.nn as nn
from torch import device


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
        params=parameters,
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


def get_embeddings():
    pass
