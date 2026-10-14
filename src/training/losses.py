"""Loss functions and class-balancing utilities for triclass classification."""

from collections.abc import Sequence

import torch
from torch import nn

from .config import TrainingConfig


def balanced_class_weights(class_counts: Sequence[int]) -> torch.Tensor:
    """Compute inverse-frequency weights normalized around one."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("class_counts must be a one-dimensional sequence with at least two classes")
    if torch.any(counts <= 0):
        raise ValueError("Every class must have at least one training sample")
    return counts.sum() / (counts.numel() * counts)


def class_weights_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Derive balanced weights from labels in the training partition only."""
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("labels must use an integer dtype")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(f"labels must be in the interval [0, {num_classes - 1}]")
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes)
    return balanced_class_weights(counts.tolist())


def build_classification_loss(
    config: TrainingConfig,
    *,
    device: torch.device | str | None = None,
) -> nn.CrossEntropyLoss:
    """Create the configured cross-entropy objective."""
    weights = None
    if config.class_weights is not None:
        weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
