"""Tests for the S9 future-training components of deliverable E4."""

from pathlib import Path

import pytest
import torch

from src.model import ViTConfig, VisionTransformer
from src.training import (
    TrainingConfig,
    balanced_class_weights,
    build_classification_loss,
    build_optimizer,
    class_weights_from_labels,
    train_batch,
)


def small_model() -> VisionTransformer:
    config = ViTConfig(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    return VisionTransformer(config)


def test_training_config_loads_project_file():
    path = Path(__file__).resolve().parents[1] / "configs" / "e4_training.json"
    config = TrainingConfig.load(path)
    assert config.optimizer == "adamw"
    assert config.num_classes == 3
    assert config.class_weights is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("label_smoothing", 1.0),
        ("gradient_clip_norm", 0.0),
    ],
)
def test_training_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


def test_balanced_class_weights():
    weights = balanced_class_weights([60, 30, 10])
    expected = torch.tensor([100 / 180, 100 / 90, 100 / 30])
    assert torch.allclose(weights, expected)


def test_balanced_class_weights_rejects_missing_class():
    with pytest.raises(ValueError, match="at least one"):
        balanced_class_weights([20, 0, 5])


def test_class_weights_from_labels():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    weights = class_weights_from_labels(labels, num_classes=3)
    assert torch.allclose(weights, balanced_class_weights([3, 2, 1]))


def test_classification_loss_uses_configured_weights():
    config = TrainingConfig(class_weights=(0.5, 1.0, 2.0), label_smoothing=0.1)
    criterion = build_classification_loss(config)
    assert torch.equal(criterion.weight, torch.tensor([0.5, 1.0, 2.0]))
    assert criterion.label_smoothing == 0.1


def test_optimizer_covers_each_trainable_parameter_once():
    model = small_model()
    optimizer = build_optimizer(model, TrainingConfig())
    optimized_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(optimized_ids) == len(set(optimized_ids))
    assert set(optimized_ids) == set(expected_ids)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.05}


def test_train_batch_updates_model_and_returns_triclass_output():
    torch.manual_seed(7)
    model = small_model()
    config = TrainingConfig(class_weights=(1.0, 1.5, 2.0))
    criterion = build_classification_loss(config)
    optimizer = build_optimizer(model, config)
    inputs = torch.randn(3, 3, 32, 32)
    targets = torch.tensor([0, 1, 2])
    before = model.classifier.head.weight.detach().clone()

    result = train_batch(
        model,
        inputs,
        targets,
        criterion,
        optimizer,
        gradient_clip_norm=config.gradient_clip_norm,
    )

    assert result.loss > 0
    assert result.logits.shape == (3, 3)
    assert result.probabilities.shape == (3, 3)
    assert torch.allclose(result.probabilities.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert result.predictions.shape == (3,)
    assert result.gradient_norm is not None
    assert not torch.equal(before, model.classifier.head.weight)


def test_train_batch_rejects_target_batch_mismatch():
    model = small_model()
    config = TrainingConfig()
    with pytest.raises(ValueError, match="match"):
        train_batch(
            model,
            torch.randn(2, 3, 32, 32),
            torch.tensor([0]),
            build_classification_loss(config),
            build_optimizer(model, config),
        )
