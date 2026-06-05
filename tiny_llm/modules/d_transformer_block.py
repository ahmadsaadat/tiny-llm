import torch
import torch.nn.functional as F

from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.utils import (
    layer_norm,
)


class TransformerBlock:
    def __init__(self, table, dimension_attention, device):
        self.table: Table = table
        self.dimension_attention = dimension_attention
        self.device = device

    def forward(self, Z):
        # normalize
        Z_norm = layer_norm(
            Z,
            self.table.table_norm1_gamma,
            self.table.table_norm1_beta,
        )

        # self-attention
        Q = Z_norm @ self.table.table_W_Q
        K = Z_norm @ self.table.table_W_K
        V = Z_norm @ self.table.table_W_V

        # attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / torch.sqrt(
            torch.tensor(self.dimension_attention, device=self.device)
        )

        # causal mask
        sequence_length = Z.shape[1]

        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=self.device),
            diagonal=1,
        )

        scores = scores.masked_fill(mask == 1, float("-inf"))

        # softmax
        attention_table = F.softmax(scores, dim=-1)

        # context mixing
        attention_output = attention_table @ V

        # output projection
        attention_output = attention_output @ self.table.table_W_O

        # residual 1
        Z = Z + attention_output

        # normalize
        Z_norm = layer_norm(
            Z,
            self.table.table_norm2_gamma,
            self.table.table_norm2_beta,
        )

        # feed forward
        hidden = Z_norm @ self.table.table_W1 + self.table.table_b1
        hidden = F.gelu(hidden)
        ffn_output = hidden @ self.table.table_W2 + self.table.table_b2

        # residual 2
        Z = Z + ffn_output

        return Z
