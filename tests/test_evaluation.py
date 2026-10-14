"""Unit tests for multiclass evaluation, confusion matrix, and metrics."""

import numpy as np
import pytest
import torch
from torch import nn

from src.data.dataset import SyntheticMRIDataset
from src.data.dataloader import create_mri_dataloader
from src.training.evaluation import (
    DEFAULT_CLASS_NAMES,
    EvaluationMetrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    evaluate_model,
    evaluate_predictions,
)


class DummyModel(nn.Module):
    """Deterministic dummy model producing constant logits."""

    def __init__(self, logits_value: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=224, stride=1)
        # Fix weights
        with torch.no_grad():
            self.conv.weight.fill_(0.0)
            self.conv.bias.fill_(logits_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.squeeze(-1).squeeze(-1)  # (B, 3)


class TestConfusionMatrix:

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        expected = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        np.testing.assert_array_equal(cm, expected)

    def test_imperfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm[0, 0] == 1 and cm[0, 1] == 1
        assert cm[1, 1] == 1 and cm[1, 2] == 1
        assert cm[2, 2] == 1 and cm[2, 0] == 1

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            compute_confusion_matrix(np.array([0, 3]), np.array([0, 1]), num_classes=3)


class TestPerClassMetrics:

    def test_metrics_calculation(self):
        # 3 classes, each has 2 samples
        cm = np.array([
            [2, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ])
        (
            per_class,
            acc,
            bal_acc,
            macro_prec,
            macro_rec,
            macro_spec,
            macro_f1,
        ) = compute_per_class_metrics(cm, class_names=("CN", "MCI", "AD"))

        assert pytest.approx(acc) == 4 / 6
        assert per_class["CN"]["recall"] == 1.0  # 2 / 2
        assert per_class["CN"]["precision"] == 2 / 3  # 2 / (2 + 1)
        assert per_class["MCI"]["recall"] == 0.5  # 1 / 2
        assert per_class["AD"]["recall"] == 0.5  # 1 / 2

    def test_empty_confusion_matrix(self):
        cm = np.zeros((3, 3), dtype=np.int64)
        per_class, acc, bal_acc, p, r, s, f1 = compute_per_class_metrics(cm)
        assert acc == 0.0
        assert f1 == 0.0
        assert per_class["CN"]["support"] == 0


class TestEvaluatePredictions:

    def test_evaluate_predictions_container(self):
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 1, 0]
        res = evaluate_predictions(y_true, y_pred, loss=0.45)
        assert isinstance(res, EvaluationMetrics)
        assert res.total_samples == 4
        assert res.loss == 0.45
        assert res.accuracy == 0.75
        assert "CN" in res.per_class
        assert len(res.summary()) > 100

    def test_serialization(self, tmp_path):
        res = evaluate_predictions([0, 1, 2], [0, 1, 2], loss=0.1)
        json_path = tmp_path / "metrics.json"
        res.to_json(json_path)
        assert json_path.is_file()


class TestEvaluateModel:

    def test_model_evaluation_loop(self):
        model = DummyModel(logits_value=0.5)
        ds = SyntheticMRIDataset(num_samples=8, seed=42)
        loader = create_mri_dataloader(ds, batch_size=4, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate_model(model, loader, criterion=criterion)
        assert metrics.total_samples == 8
        assert metrics.loss is not None
        assert 0.0 <= metrics.accuracy <= 1.0
