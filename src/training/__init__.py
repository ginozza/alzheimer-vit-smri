"""Training components prepared for future ViT optimization."""

from .config import TrainingConfig
from .engine import TrainingStepResult, train_batch
from .evaluation import (
    DEFAULT_CLASS_NAMES,
    EvaluationMetrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    evaluate_model,
    evaluate_predictions,
)
from .losses import (
    balanced_class_weights,
    build_classification_loss,
    class_weights_from_labels,
)
from .optimizers import build_optimizer
from .trainer import FutureTrainer, TrainingHistory

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "EvaluationMetrics",
    "FutureTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingStepResult",
    "balanced_class_weights",
    "build_classification_loss",
    "build_optimizer",
    "class_weights_from_labels",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "evaluate_model",
    "evaluate_predictions",
    "train_batch",
]
