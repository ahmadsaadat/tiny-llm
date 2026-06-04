import torch
import torch.nn as nn
from torch import device


class Tables:
    size_vocab: int
    size_training_sequences: int
    table_embedding = {}
    table_position = {}
    table_lm_head = {}
    table_norm1_gamma = {}
    table_norm1_beta = {}
    table_norm2_gamma = {}
    table_norm2_beta = {}
    table_W1 = {}
    table_b1 = {}
    table_W2 = {}
    table_b2 = {}
    table_W_Q = {}
    table_W_K = {}
    table_W_V = {}
    table_W_O = {}

    def __init__(
        self,
        size_vocab: int,
        size_training_sequences: int,
        dimension_embedding: int = 64,
        dimension_attention: int = 64,
        device: device = torch.device("mps"),
    ):
        self.size_vocab = size_vocab
        self.size_training_sequences = size_training_sequences
        self.table_embedding = nn.Parameter(
            torch.randn(size_vocab, dimension_embedding, device=device) * 0.01
        )
        self.table_position = nn.Parameter(
            torch.randn(size_training_sequences, dimension_embedding, device=device)
            * 0.01
        )
        self.table_lm_head = nn.Parameter(
            torch.randn(dimension_embedding, size_vocab, device=device) * 0.01
        )
        # layer norm
        self.table_norm1_gamma = nn.Parameter(
            torch.ones(dimension_embedding, device=device)
        )
        self.table_norm1_beta = nn.Parameter(
            torch.zeros(dimension_embedding, device=device)
        )
        self.table_norm2_gamma = nn.Parameter(
            torch.ones(dimension_embedding, device=device)
        )
        self.table_norm2_beta = nn.Parameter(
            torch.zeros(dimension_embedding, device=device)
        )
        # feed forward tables
        self.W1 = nn.Parameter(
            torch.randn(dimension_embedding, dimension_embedding * 4, device=device)
            * 0.01
        )
        self.b1 = nn.Parameter(torch.zeros(dimension_embedding * 4, device=device))
        self.W2 = nn.Parameter(
            torch.randn(dimension_embedding * 4, dimension_embedding, device=device)
            * 0.01
        )
        self.b2 = nn.Parameter(torch.zeros(dimension_embedding, device=device))
        # attention tables
        self.W_Q = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.W_K = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.W_V = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.W_O = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
