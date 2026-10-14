"""Verify the S9 ViT output contract and future-training components.

This script performs structural checks with synthetic tensors only. It does not
train the project model or produce diagnostic results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import ViTConfig, VisionTransformer
from src.training import (
    TrainingConfig,
    balanced_class_weights,
    build_classification_loss,
    build_optimizer,
    train_batch,
)


def verify_full_architecture() -> None:
    config = ViTConfig.load("configs/e3_architecture.json")
    model = VisionTransformer(config)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 224, 224))

    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    assert model.count_parameters() == 85_800_963
    print("[OK] ViT-B/16: entrada (1, 3, 224, 224) -> logits (1, 3)")
    print(f"[OK] Parametros entrenables: {model.count_parameters():,}")


def verify_future_training_components() -> None:
    model_config = ViTConfig(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    model = VisionTransformer(model_config)
    weights = balanced_class_weights([60, 30, 10])
    training_config = TrainingConfig(class_weights=tuple(float(value) for value in weights))
    criterion = build_classification_loss(training_config)
    optimizer = build_optimizer(model, training_config)
    result = train_batch(
        model,
        torch.randn(3, 3, 32, 32),
        torch.tensor([0, 1, 2]),
        criterion,
        optimizer,
        gradient_clip_norm=training_config.gradient_clip_norm,
    )

    assert result.logits.shape == (3, 3)
    assert torch.allclose(result.probabilities.sum(dim=-1), torch.ones(3), atol=1e-6)
    print(f"[OK] Pesos balanceados de prueba: {[round(value, 4) for value in weights.tolist()]}")
    print(f"[OK] Paso sintetico forward/backward: perdida={result.loss:.6f}")
    print("[OK] AdamW, entropia cruzada ponderada y clipping de gradiente operativos")


def main() -> None:
    print("VERIFICACION S9 - ARQUITECTURA TRICLASE Y ENTRENAMIENTO FUTURO")
    verify_full_architecture()
    verify_future_training_components()
    print("RESULTADO: S9 verificada sin entrenamiento ni datos clinicos reales")


if __name__ == "__main__":
    main()
