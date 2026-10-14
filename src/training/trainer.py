"""Training loop orchestration and checkpointing for future model optimization.

Provides FutureTrainer to coordinate multi-epoch training, validation passes,
metrics tracking, and checkpoint persistence without clinical claims.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .engine import train_batch
from .evaluation import EvaluationMetrics, evaluate_model


@dataclass
class TrainingHistory:
    """Historical record of training and validation runs across epochs."""

    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_macro_f1: List[float] = field(default_factory=list)

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        val_f1: Optional[float] = None,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_accuracy.append(val_acc)
        if val_f1 is not None:
            self.val_macro_f1.append(val_f1)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)


class FutureTrainer:
    """Orchestrates training and validation passes for the ViT-B/16 system.

    Designed to verify pipeline integration, gradient backpropagation,
    and checkpoint saving/loading ahead of final experimental runs.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        config: Optional[TrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.device = torch.device(device if device is not None else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.lr_scheduler = lr_scheduler
        self.history = TrainingHistory()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch across the provided DataLoader."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            if isinstance(batch, dict) and "images" in batch and "labels" in batch:
                inputs = batch["images"].to(self.device)
                targets = batch["labels"].to(self.device)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                raise TypeError(f"Unsupported batch format: {type(batch)}")

            step_res = train_batch(
                model=self.model,
                inputs=inputs,
                targets=targets,
                criterion=self.criterion,
                optimizer=self.optimizer,
                gradient_clip_norm=self.config.gradient_clip_norm,
            )
            total_loss += step_res.loss
            batch_count += 1

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss / max(batch_count, 1)

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[Optional[float], EvaluationMetrics]:
        """Run one evaluation pass across the provided validation DataLoader."""
        metrics = evaluate_model(
            model=self.model,
            dataloader=dataloader,
            criterion=self.criterion,
            device=self.device,
        )
        return metrics.loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> TrainingHistory:
        """Execute a controlled training loop across multiple epochs."""
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            val_loss = None
            val_acc = None
            val_f1 = None
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                val_acc = val_metrics.accuracy
                val_f1 = val_metrics.macro_f1

            self.history.record(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1=val_f1,
            )

        return self.history

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist model and optimizer states to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": self.config.as_dict(),
            "history": self.history.to_dict(),
            "metrics": metrics or {},
        }
        torch.save(payload, target_path)
        return target_path

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Restore model and optimizer states from disk."""
        target_path = Path(path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {target_path}")

        payload = torch.load(target_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload
