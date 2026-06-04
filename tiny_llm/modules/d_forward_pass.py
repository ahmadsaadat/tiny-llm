import torch

from tiny_llm.modules.e_transformer_block import transformer_block


# 4
def forward_pass(
    batch_input_ids,
    weights,
    sequence_length,
    attention_dimension,
    n_heads,
    device,
):
    """AKA forward pass"""
    # embeddings
    X = weights["embedding_table"][batch_input_ids]

    # position embeddings
    positions = torch.arange(
        sequence_length,
        device=device,
    ).unsqueeze(0)

    Z = X + weights["position_embedding_table"][positions]

    # transformer block
    Z = transformer_block(
        Z=Z,
        weights=weights,
        sequence_length=sequence_length,
        attention_dimension=attention_dimension,
        n_heads=n_heads,
        device=device,
    )

    # lm head
    last_token_vector = Z[:, -1]

    logits = last_token_vector @ weights["lm_head"]

    return logits
