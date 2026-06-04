import torch
from torch import device


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return gamma * ((x - mean) / (std + eps)) + beta


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
