import torch
import torch.nn.functional as F
from torch import device

from tiny_llm.modules.utils import (
    layer_norm,
)


def transformer_block(
    Z,
    weights,
    sequence_length,
    attention_dimension,
    n_heads: int,
    device: device,
):
    """AKA context builder"""
    head_dimension = attention_dimension // n_heads

    # 11.7 normalize
    Z_norm = layer_norm(
        Z,
        weights["norm1_gamma"],
        weights["norm1_beta"],
    )

    # 11.8 self-attention
    Q = Z_norm @ weights["W_Q"]
    K = Z_norm @ weights["W_K"]
    V = Z_norm @ weights["W_V"]

    current_batch_size = Z.shape[0]

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
    attention_output = attention_output @ weights["W_O"]

    # residual connection 1
    Z = Z + attention_output

    # layernorm 2
    Z_norm = layer_norm(
        Z,
        weights["norm2_gamma"],
        weights["norm2_beta"],
    )

    # feed forward network
    hidden = Z_norm @ weights["W1"] + weights["b1"]
    hidden = F.gelu(hidden)
    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # residual connection 2
    Z = Z + ffn_output

    return Z
