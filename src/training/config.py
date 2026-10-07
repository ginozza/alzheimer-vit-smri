"""Configuration for the future ViT training workflow."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for reproducible future training without starting a run."""

    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = 1.0
    num_classes: int = 3
    class_weights: Tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.optimizer.lower() != "adamw":
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2 or any(beta < 0 or beta >= 1 for beta in self.betas):
            raise ValueError("betas must contain two values in the interval [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in the interval [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be greater than zero when provided")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if self.class_weights is not None:
            if len(self.class_weights) != self.num_classes:
                raise ValueError("class_weights length must match num_classes")
            if any(weight <= 0 for weight in self.class_weights):
                raise ValueError("class_weights values must be greater than zero")

    def as_dict(self) -> dict:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["betas"] = list(self.betas)
        if self.class_weights is not None:
            result["class_weights"] = list(self.class_weights)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """Load and validate a training configuration from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "betas" in raw:
            raw["betas"] = tuple(raw["betas"])
        if raw.get("class_weights") is not None:
            raw["class_weights"] = tuple(raw["class_weights"])
        return cls(**raw)
