import torch
from tokenizers import Tokenizer
from torch import device

from tiny_llm.modules.utils import (
    create_attention_tables,
    create_embedding_tables,
    create_feedforward_tables,
    create_layernorm_tables,
    create_lmhead_tables,
)


def create_weights(
    tokenizer: Tokenizer,
    sequences: list[tuple],
    embedding_dimension: int = 64,
    attention_dimension: int = 64,
    device: device = torch.device("mps"),
):
    print("Using Apple GPU:", device)

    embedding_table, position_embedding_table = create_embedding_tables(
        embedding_dimension,
        tokenizer.get_vocab_size(),
        len(sequences),
        device,
    )

    # 6. LM head tables
    lm_head = create_lmhead_tables(
        embedding_dimension,
        tokenizer.get_vocab_size(),
        device,
    )

    # 7. Layer Norm tables
    norm1_gamma, norm1_beta, norm2_gamma, norm2_beta = create_layernorm_tables(
        embedding_dimension,
        device,
    )

    # 8. Feed Forward tables
    W1, b1, W2, b2 = create_feedforward_tables(
        embedding_dimension,
        device,
    )

    # 9. Attention tables
    W_Q, W_K, W_V, W_O = create_attention_tables(
        embedding_dimension,
        attention_dimension,
        device,
    )

    return {
        "embedding_table": embedding_table,
        "position_embedding_table": position_embedding_table,
        "W_Q": W_Q,
        "W_K": W_K,
        "W_V": W_V,
        "W_O": W_O,
        "norm1_gamma": norm1_gamma,
        "norm1_beta": norm1_beta,
        "norm2_gamma": norm2_gamma,
        "norm2_beta": norm2_beta,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "lm_head": lm_head,
    }
