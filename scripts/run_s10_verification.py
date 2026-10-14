"""Verify the S10 integrated system components for EDT 5.0.

Tests data loading, model integration, future training loop, evaluation metrics,
and inference pipeline using synthetic tensors. Does not train project weights
or claim clinical validity.
"""

from pathlib import Path
import sys
import tempfile

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.data.dataset import SyntheticMRIDataset
from src.data.dataloader import create_mri_dataloader
from src.inference import InferencePipeline, PredictionResult
from src.model import ViTConfig, VisionTransformer
from src.training import (
    FutureTrainer,
    TrainingConfig,
    balanced_class_weights,
    build_classification_loss,
    build_optimizer,
    evaluate_model,
)


def verify_data_loading() -> None:
    print("\n[PASO 1] Verificando modulo de datos y DataLoader (EDT 5.1)...")
    dataset = SyntheticMRIDataset(num_samples=16, num_classes=3, seed=42)
    assert len(dataset) == 16
    loader = create_mri_dataloader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    assert "images" in batch and "labels" in batch
    assert batch["images"].shape == (4, 3, 224, 224)
    assert batch["images"].dtype == torch.float32
    assert batch["labels"].shape == (4,)
    assert batch["labels"].dtype == torch.int64
    assert len(batch["subject_ids"]) == 4
    print("  [OK] Dataset y DataLoader: lote (4, 3, 224, 224) float32 y etiquetas (4,) int64")


def verify_model_integration() -> None:
    print("\n[PASO 2] Verificando integracion con modelo ViT-B/16 (EDT 5.2)...")
    config = ViTConfig.load("configs/e3_architecture.json")
    model = VisionTransformer(config)
    model.eval()

    sample_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(sample_input)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    print("  [OK] Integracion ViT-B/16: lote (2, 3, 224, 224) -> logits (2, 3)")


def verify_training_and_checkpointing() -> None:
    print("\n[PASO 3] Verificando entrenamiento futuro y checkpointing (EDT 5.3)...")
    # Use compact ViT for fast verification pass
    config = ViTConfig(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    model = VisionTransformer(config)
    weights = balanced_class_weights([40, 30, 10])
    training_config = TrainingConfig(
        learning_rate=1e-3,
        class_weights=tuple(float(v) for v in weights),
    )
    criterion = build_classification_loss(training_config)
    optimizer = build_optimizer(model, training_config)

    trainer = FutureTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=training_config,
        device="cpu",
    )

    train_ds = SyntheticMRIDataset(num_samples=12, image_shape=(3, 32, 32), seed=10)
    val_ds = SyntheticMRIDataset(num_samples=6, image_shape=(3, 32, 32), seed=20)
    train_loader = create_mri_dataloader(train_ds, batch_size=4, shuffle=True)
    val_loader = create_mri_dataloader(val_ds, batch_size=3, shuffle=False)

    history = trainer.fit(train_loader, val_loader, epochs=2)
    assert len(history.epochs) == 2
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model_ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=2, metrics={"val_acc": history.val_accuracy[-1]})
        assert ckpt_path.is_file()

        # Test reload
        restored_model = VisionTransformer(config)
        restored_opt = build_optimizer(restored_model, training_config)
        restored_trainer = FutureTrainer(
            model=restored_model,
            criterion=criterion,
            optimizer=restored_opt,
            config=training_config,
            device="cpu",
        )
        payload = restored_trainer.load_checkpoint(ckpt_path)
        assert payload["epoch"] == 2
    print(f"  [OK] Multi-epoch loop ejecutado: epocas={len(history.epochs)}, loss_final={history.train_loss[-1]:.4f}")
    print("  [OK] Checkpoint guardado y restaurado exitosamente")


def verify_evaluation_metrics() -> None:
    print("\n[PASO 4] Verificando modulo de evaluacion multiclase (EDT 5.4)...")
    config = ViTConfig(
        image_size=32,
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    model = VisionTransformer(config)
    val_ds = SyntheticMRIDataset(num_samples=12, image_shape=(3, 32, 32), seed=30)
    val_loader = create_mri_dataloader(val_ds, batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(model, val_loader, criterion=criterion)
    assert metrics.total_samples == 12
    assert metrics.loss is not None
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert len(metrics.confusion_matrix) == 3
    for name in ("CN", "MCI", "AD"):
        assert name in metrics.per_class

    print(f"  [OK] Evaluacion multiclase: Acc={metrics.accuracy*100:.1f}%, F1_Macro={metrics.macro_f1*100:.1f}%")
    print(f"  [OK] Matriz de confusion 3x3 generada:\n       {metrics.confusion_matrix}")


def verify_inference_pipeline() -> None:
    print("\n[PASO 5] Verificando pipeline de inferencia (EDT 5.5)...")
    config = ViTConfig.load("configs/e3_architecture.json")
    model = VisionTransformer(config)
    pipeline = InferencePipeline(model=model, device="cpu")

    # Single tensor inference
    single_tensor = torch.randn(3, 224, 224)
    pred_single = pipeline.predict_tensor(single_tensor, return_attention=True)
    assert isinstance(pred_single, PredictionResult)
    assert pred_single.predicted_class_id in (0, 1, 2)
    assert pred_single.predicted_class_name in ("CN", "MCI", "AD")
    assert pytest_approx_sum(pred_single.probabilities.values(), 1.0)
    assert pred_single.attention_maps is not None
    assert len(pred_single.attention_maps) == 12

    # Batch tensor inference
    batch_tensor = torch.randn(3, 3, 224, 224)
    preds_batch = pipeline.predict_batch(batch_tensor)
    assert len(preds_batch) == 3
    for p in preds_batch:
        assert p.predicted_class_name in ("CN", "MCI", "AD")

    print(f"  [OK] Inferencia unitaria: Clase={pred_single.predicted_class_name}, Confianza={pred_single.confidence:.4f}")
    print(f"  [OK] Probabilidades: {pred_single.probabilities}")
    print(f"  [OK] Mapas de atencion capturados en inferencia: {len(pred_single.attention_maps)} capas")
    print(f"  [OK] Inferencia por lote ({len(preds_batch)} muestras) completada exitosamente")


def pytest_approx_sum(vals, target, tol=1e-5):
    return abs(sum(vals) - target) < tol


def main() -> None:
    print("=" * 80)
    print("  VERIFICACION INTEGRAL S10: CODIGO INTEGRADO DE COMPONENTES (EDT 5.0)")
    print("  Seminario III (2026-II) - Universidad del Magdalena")
    print("  Responsables: Malak Sanchez y Juan Simancas")
    print("=" * 80)

    verify_data_loading()
    verify_model_integration()
    verify_training_and_checkpointing()
    verify_evaluation_metrics()
    verify_inference_pipeline()

    print("\n" + "=" * 80)
    print("  RESULTADO: S10 VERIFICADA AL 100 %")
    print("  Componentes integrados: Dataset, DataLoader, ViT-B/16, Trainer, Evaluator, Predictor.")
    print("  Restriccion academica preservada: Sin entrenamiento ni pretension de validez clinica.")
    print("=" * 80)


if __name__ == "__main__":
    main()
