"""Multiclass diagnostic evaluation metrics for triclass Alzheimer classification.

Computes confusion matrix, macro precision/recall/F1, specificity, balanced accuracy,
and per-class statistics with zero-division protection.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


DEFAULT_CLASS_NAMES: Tuple[str, ...] = ("CN", "MCI", "AD")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Standardized multiclass evaluation metrics container."""

    total_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_specificity: float
    macro_f1: float
    confusion_matrix: List[List[int]]
    per_class: Dict[str, Dict[str, Union[float, int]]]
    loss: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Evaluation Metrics Summary",
            "=" * 60,
            f"  Total samples:        {self.total_samples}",
            f"  Loss:                 {self.loss:.4f}" if self.loss is not None else "  Loss:                 N/A",
            f"  Accuracy:             {self.accuracy * 100:.2f} %",
            f"  Balanced Accuracy:    {self.balanced_accuracy * 100:.2f} %",
            f"  Macro Precision:      {self.macro_precision * 100:.2f} %",
            f"  Macro Recall (Sens):  {self.macro_recall * 100:.2f} %",
            f"  Macro Specificity:    {self.macro_specificity * 100:.2f} %",
            f"  Macro F1-Score:       {self.macro_f1 * 100:.2f} %",
            "-" * 60,
            "  Per-class metrics:",
        ]
        for name, m in self.per_class.items():
            lines.append(
                f"    [{name}] Prec: {m['precision']*100:.1f}%, "
                f"Rec: {m['recall']*100:.1f}%, "
                f"Spec: {m['specificity']*100:.1f}%, "
                f"F1: {m['f1']*100:.1f}%, "
                f"Support: {m['support']}"
            )
        lines.append("-" * 60)
        lines.append(f"  Confusion Matrix (rows=True, cols=Pred):")
        for row in self.confusion_matrix:
            lines.append(f"    {row}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """Compute (num_classes, num_classes) integer confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
        else:
            raise ValueError(f"Label out of bounds: true={t}, pred={p}, num_classes={num_classes}")
    return cm


def compute_per_class_metrics(
    cm: np.ndarray,
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> Tuple[Dict[str, Dict[str, Union[float, int]]], float, float, float, float, float, float]:
    """Compute per-class and macro averaged metrics from confusion matrix."""
    num_classes = cm.shape[0]
    total_samples = int(np.sum(cm))

    if total_samples == 0:
        empty_per_class = {
            name: {"precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0, "support": 0}
            for name in class_names
        }
        return empty_per_class, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    per_class = {}
    precisions = []
    recalls = []
    specificities = []
    f1s = []

    for i in range(num_classes):
        name = class_names[i]
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total_samples - (tp + fn + fp))
        support = int(tp + fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[name] = {
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "f1": float(f1),
            "support": support,
        }

        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)
        f1s.append(f1)

    accuracy = float(np.trace(cm) / total_samples)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_specificity = float(np.mean(specificities))
    macro_f1 = float(np.mean(f1s))
    balanced_accuracy = macro_recall  # Balanced accuracy is macro recall

    return (
        per_class,
        accuracy,
        balanced_accuracy,
        macro_precision,
        macro_recall,
        macro_specificity,
        macro_f1,
    )


def evaluate_predictions(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    loss: Optional[float] = None,
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> EvaluationMetrics:
    """Evaluate arrays of ground truth and predicted labels."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    (
        per_class,
        acc,
        bal_acc,
        macro_prec,
        macro_rec,
        macro_spec,
        macro_f1,
    ) = compute_per_class_metrics(cm, class_names=class_names)

    return EvaluationMetrics(
        total_samples=len(y_true),
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_specificity=macro_spec,
        macro_f1=macro_f1,
        confusion_matrix=cm.tolist(),
        per_class=per_class,
        loss=loss,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> EvaluationMetrics:
    """Execute evaluation loop over a DataLoader.

    Args:
        model: VisionTransformer or compatible model.
        dataloader: DataLoader yielding batches with 'images' and 'labels'.
        criterion: Optional loss module to compute mean validation loss.
        device: Torch device (defaults to CPU).
        class_names: Tuple of diagnostic class names.

    Returns:
        EvaluationMetrics instance with complete metrics and confusion matrix.
    """
    dev = torch.device(device if device is not None else "cpu")
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict) and "images" in batch and "labels" in batch:
                images = batch["images"].to(dev)
                labels = batch["labels"].to(dev)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images = batch[0].to(dev)
                labels = batch[1].to(dev)
            else:
                raise TypeError(f"Unsupported batch format from DataLoader: {type(batch)}")

            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, labels)
                if torch.isfinite(loss):
                    total_loss += float(loss.detach().cpu())
                    num_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    mean_loss = (total_loss / num_batches) if num_batches > 0 else None
    return evaluate_predictions(
        y_true=all_targets,
        y_pred=all_preds,
        loss=mean_loss,
        class_names=class_names,
    )
