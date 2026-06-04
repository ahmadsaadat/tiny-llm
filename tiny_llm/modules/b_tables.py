import torch
import torch.nn as nn
from torch import device


class Tables:
    vocab_size: int
    sequence_length: int
    # embedding tables
    table_embedding = {}
    table_position = {}
    table_lm_head = {}
    # layer norm tables
    table_norm1_gamma = {}
    table_norm1_beta = {}
    table_norm2_gamma = {}
    table_norm2_beta = {}
    # feed forward tables
    table_W1 = {}
    table_b1 = {}
    table_W2 = {}
    table_b2 = {}
    # attention tables
    table_W_Q = {}
    table_W_K = {}
    table_W_V = {}
    table_W_O = {}

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        dimension_embedding: int = 64,
        dimension_attention: int = 64,
        device: device = torch.device("mps"),
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.table_embedding = nn.Parameter(
            torch.randn(vocab_size, dimension_embedding, device=device) * 0.01
        )
        self.table_position = nn.Parameter(
            torch.randn(sequence_length, dimension_embedding, device=device) * 0.01
        )
        self.table_lm_head = nn.Parameter(
            torch.randn(dimension_embedding, vocab_size, device=device) * 0.01
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
        self.table_W1 = nn.Parameter(
            torch.randn(dimension_embedding, dimension_embedding * 4, device=device)
            * 0.01
        )
        self.table_b1 = nn.Parameter(
            torch.zeros(dimension_embedding * 4, device=device)
        )
        self.table_W2 = nn.Parameter(
            torch.randn(dimension_embedding * 4, dimension_embedding, device=device)
            * 0.01
        )
        self.table_b2 = nn.Parameter(torch.zeros(dimension_embedding, device=device))
        # attention tables
        self.table_W_Q = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.table_W_K = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.table_W_V = nn.Parameter(
            torch.randn(dimension_embedding, dimension_attention, device=device) * 0.01
        )
        self.table_W_O = nn.Parameter(
            torch.randn(dimension_attention, dimension_embedding, device=device) * 0.01
        )

    @property
    def parameters(self):
        return [
            self.table_embedding,
            self.table_position,
            self.table_lm_head,
            self.table_norm1_gamma,
            self.table_norm1_beta,
            self.table_norm2_gamma,
            self.table_norm2_beta,
            self.table_W1,
            self.table_b1,
            self.table_W2,
            self.table_b2,
            self.table_W_Q,
            self.table_W_K,
            self.table_W_V,
            self.table_W_O,
        ]
