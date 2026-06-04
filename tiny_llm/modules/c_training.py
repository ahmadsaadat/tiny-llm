import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from tiny_llm.modules.d_forward_pass import forward_pass
from tiny_llm.modules.utils import get_batch, get_batch_inputs, get_batch_targets


def training_loop(
    sequences,
    weights,
    optimizer,
    sequence_length,
    attention_dimension,
    n_heads,
    batch_size,
    epochs,
    device=torch.device("mps"),
):
    for epoch in range(epochs):
        total_loss = 0

        random.shuffle(sequences)

        progress_bar = tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
        )

        for step in progress_bar:
            batch = get_batch(
                training_sequences=sequences,
                step=step,
                batch_size=batch_size,
            )

            batch_input_ids = get_batch_inputs(
                batch=batch,
                device=device,
            )

            batch_target_ids = get_batch_targets(
                batch=batch,
                device=device,
            )

            logits = forward_pass(
                batch_input_ids=batch_input_ids,
                weights=weights,
                sequence_length=sequence_length,
                attention_dimension=attention_dimension,
                n_heads=n_heads,
                device=device,
            )

            loss = F.cross_entropy(
                logits,
                batch_target_ids,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
            )

        avg_loss = total_loss / len(progress_bar)

        print(f"\nEpoch {epoch + 1}/{epochs} | avg loss = {avg_loss:.4f}")
