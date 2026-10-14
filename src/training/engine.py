"""Minimal batch-level engine for structural training verification."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class TrainingStepResult:
    """Detached outputs from one optimization step."""

    loss: float
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    gradient_norm: float | None


def train_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: Optimizer,
    *,
    gradient_clip_norm: float | None = None,
) -> TrainingStepResult:
    """Execute one training batch for synthetic or future real-data workflows."""
    if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
        raise ValueError("targets must have shape (B,) and match the input batch size")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss detected")
    loss.backward()

    gradient_norm = None
    if gradient_clip_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        gradient_norm = float(norm.detach().cpu())

    optimizer.step()
    detached_logits = logits.detach()
    probabilities = torch.softmax(detached_logits, dim=-1)
    return TrainingStepResult(
        loss=float(loss.detach().cpu()),
        logits=detached_logits,
        probabilities=probabilities,
        predictions=probabilities.argmax(dim=-1),
        gradient_norm=gradient_norm,
    )
