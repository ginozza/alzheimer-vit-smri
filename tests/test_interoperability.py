"""Interoperability and end-to-end integration tests for Deliverable E4 (Semana 11).

Validates the complete execution contract across all subsystems:
- Data preprocessors (E1/E2) -> Dataset & DataLoader
- DataLoader -> ViT-B/16 architecture (E3)
- ViT-B/16 -> Training loop & loss calculation (E4)
- ViT-B/16 -> Evaluation metrics & confusion matrix (E4)
- ViT-B/16 -> Diagnostic inference pipeline & attention rollout bridge (E4/E5)
- Cross-configuration consistency across configs/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    MRI25DDataset,
    SyntheticMRIDataset,
)
from src.data.dataloader import create_mri_dataloader
from src.data.preprocessing import PreprocessingConfig
from src.data.slice_representation import CanonicalTripletExtractor
from src.data.subject_split import SubjectSplitter
from src.inference.predictor import InferencePipeline, PredictionResult
from src.model.config import ViTConfig
from src.model.vit import VisionTransformer
from src.training.config import TrainingConfig
from src.training.evaluation import EvaluationMetrics, evaluate_model
from src.training.losses import balanced_class_weights, build_classification_loss
from src.training.optimizers import build_optimizer
from src.training.trainer import FutureTrainer


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class TestCrossConfigurationConsistency:
    """Verifies that all external JSON configuration files are consistent."""

    def test_all_configs_exist(self):
        expected_files = [
            "e2.json",
            "e3_architecture.json",
            "e4_training.json",
            "e4_dataset.json",
            "e4_inference.json",
        ]
        for fname in expected_files:
            p = CONFIGS_DIR / fname
            assert p.is_file(), f"Missing configuration file: {p}"

    def test_dimensions_and_classes_aligned(self):
        vit_cfg = ViTConfig.load(CONFIGS_DIR / "e3_architecture.json")
        train_cfg = TrainingConfig.load(CONFIGS_DIR / "e4_training.json")
        dataset_cfg = json.loads((CONFIGS_DIR / "e4_dataset.json").read_text(encoding="utf-8"))
        inference_cfg = json.loads((CONFIGS_DIR / "e4_inference.json").read_text(encoding="utf-8"))

        # Image size & channels
        assert vit_cfg.image_size == 224
        assert vit_cfg.in_channels == 3
        assert dataset_cfg["target_shape"] == [3, 224, 224]

        # Number of classes
        assert vit_cfg.num_classes == 3
        assert train_cfg.num_classes == 3
        assert dataset_cfg["num_classes"] == 3
        assert inference_cfg["num_classes"] == 3

        # Class naming order
        assert vit_cfg.class_names == ("CN", "MCI", "AD")
        assert tuple(dataset_cfg["class_names"]) == ("CN", "MCI", "AD")
        assert tuple(inference_cfg["class_names"]) == ("CN", "MCI", "AD")


class TestEndToEndInteroperabilityPipeline:
    """Tests the full chain: Data Preprocessing -> Dataset -> ViT -> Train -> Eval -> Infer."""

    def test_full_synthetic_pipeline_execution(self, tmp_path):
        # 1. Generate 2.5D slices and save to disk simulating E2 preprocessing output
        extractor = CanonicalTripletExtractor()
        batch_x, batch_y = extractor.generate_synthetic_batch(batch_size=9)
        tensor_dir = tmp_path / "e2_outputs"
        tensor_dir.mkdir()

        records = []
        for i in range(len(batch_x)):
            t_path = tensor_dir / f"slice_{i:03d}.npy"
            arr = batch_x[i] if isinstance(batch_x[i], np.ndarray) else batch_x[i].numpy()
            np.save(t_path, arr)
            diag_idx = int(batch_y[i])
            records.append({
                "Subject_ID": f"SUBJ_{i // 3:03d}",
                "Session_ID": f"SES_{i % 3:02d}",
                "Cohort_Source": "INTEROP-SYNTH",
                "Diagnosis_Numeric": diag_idx,
                "Diagnosis_Class": CLASS_NAMES[diag_idx],
                "Output_Tensor_Path": str(t_path),
            })

        manifest_df = pd.DataFrame(records)
        manifest_csv = tmp_path / "outputs.csv"
        manifest_df.to_csv(manifest_csv, index=False)

        # 2. Ingest via MRI25DDataset & DataLoader (EDT 5.1)
        dataset = MRI25DDataset(manifest=manifest_csv)
        assert len(dataset) == 9
        loader = create_mri_dataloader(dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        assert batch["images"].shape == (3, 3, 224, 224)
        assert batch["labels"].shape == (3,)

        # 3. Model instantiation (EDT 5.2 / E3)
        vit_cfg = ViTConfig.load(CONFIGS_DIR / "e3_architecture.json")
        model = VisionTransformer(vit_cfg)
        assert model.count_parameters() == 85_800_963

        # 4. Training step execution (EDT 5.3)
        class_counts = [3, 3, 3]
        weights = balanced_class_weights(class_counts)
        train_cfg = TrainingConfig(
            learning_rate=1e-4,
            class_weights=tuple(float(w) for w in weights),
        )
        criterion = build_classification_loss(train_cfg)
        optimizer = build_optimizer(model, train_cfg)
        trainer = FutureTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=train_cfg,
            device="cpu",
        )

        history = trainer.fit(train_loader=loader, val_loader=loader, epochs=1)
        assert len(history.train_loss) == 1
        assert history.train_loss[0] > 0.0

        # Save checkpoint
        ckpt_path = tmp_path / "interop_ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        assert ckpt_path.is_file()

        # 5. Multiclass Evaluation (EDT 5.4)
        eval_metrics = evaluate_model(model, loader, criterion=criterion)
        assert isinstance(eval_metrics, EvaluationMetrics)
        assert eval_metrics.total_samples == 9
        assert len(eval_metrics.confusion_matrix) == 3
        assert 0.0 <= eval_metrics.accuracy <= 1.0

        # 6. Diagnostic Inference Pipeline (EDT 5.5)
        inference = InferencePipeline(model=model, device="cpu")

        # Test single slice file prediction directly from E2 output
        first_tensor_file = records[0]["Output_Tensor_Path"]
        file_pred = inference.predict_file(first_tensor_file, return_attention=True)
        assert isinstance(file_pred, PredictionResult)
        assert file_pred.predicted_class_name in CLASS_NAMES
        assert 0.0 <= file_pred.confidence <= 1.0
        assert pytest.approx(sum(file_pred.probabilities.values())) == 1.0
        # Attention maps extracted across 12 layers for E5
        assert file_pred.attention_maps is not None
        assert len(file_pred.attention_maps) == 12

        # Test batch prediction
        batch_preds = inference.predict_batch(batch["images"])
        assert len(batch_preds) == 3
        for pred in batch_preds:
            assert pred.predicted_class_name in CLASS_NAMES


class TestSubjectPartitioningAndDatasetLeakage:
    """Verifies that subject-level splits feed cleanly into datasets with zero subject leakage."""

    def test_disjoint_partitions_in_datasets(self, tmp_path):
        # Create manifest with multiple sessions per subject
        records = []
        tensor_dir = tmp_path / "leakage_tensors"
        tensor_dir.mkdir()

        subjects = [f"SUBJ_{s:02d}" for s in range(30)]
        for s_idx, subj in enumerate(subjects):
            diag = s_idx % 3
            for ses in range(2):
                t_path = tensor_dir / f"{subj}_ses{ses}.npy"
                np.save(t_path, np.zeros((3, 224, 224), dtype=np.float32))
                records.append({
                    "Subject_ID": subj,
                    "Session_ID": f"MR{ses}",
                    "Cohort_Source": "AUDIT",
                    "Diagnosis_Numeric": diag,
                    "Diagnosis_Class": CLASS_NAMES[diag],
                    "Age": 72.0 + (s_idx % 5),
                    "Sex": "F" if s_idx % 2 == 0 else "M",
                    "Output_Tensor_Path": str(t_path),
                })

        df = pd.DataFrame(records)
        splitter = SubjectSplitter(random_seed=42)
        split = splitter.split_holdout(df, test_size=0.25, val_size_from_dev=0.20)

        # Datasets
        train_ds = MRI25DDataset(manifest=split.train_df)
        val_ds = MRI25DDataset(manifest=split.val_df)
        test_ds = MRI25DDataset(manifest=split.test_df)

        s_train = {train_ds[i]["subject_id"] for i in range(len(train_ds))}
        s_val = {val_ds[i]["subject_id"] for i in range(len(val_ds))}
        s_test = {test_ds[i]["subject_id"] for i in range(len(test_ds))}

        # Mathematical zero leakage assertion
        assert len(s_train.intersection(s_val)) == 0
        assert len(s_train.intersection(s_test)) == 0
        assert len(s_val.intersection(s_test)) == 0
        assert len(s_train) + len(s_val) + len(s_test) == 30
