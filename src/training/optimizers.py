"""Optimizer construction for future ViT training."""

from torch import nn
from torch.optim import AdamW, Optimizer

from .config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Build AdamW with weight decay excluded from biases and scale parameters."""
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim < 2
            or name.endswith(".bias")
            or name.endswith("cls_token")
            or name.endswith("position_embeddings")
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
    )
