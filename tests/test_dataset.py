"""Unit tests for MRI Dataset, Synthetic Dataset, and DataLoader utilities."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MRI25DDataset,
    SyntheticMRIDataset,
)
from src.data.dataloader import create_mri_dataloader, mri_collate_fn


class TestSyntheticMRIDataset:

    def test_length_and_shapes(self):
        ds = SyntheticMRIDataset(num_samples=15, num_classes=3, seed=123)
        assert len(ds) == 15
        sample = ds[0]
        assert isinstance(sample, dict)
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.int64
        assert sample["label"].item() in (0, 1, 2)
        assert sample["diagnosis_name"] in ("CN", "MCI", "AD")

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="positive"):
            SyntheticMRIDataset(num_samples=0)
        with pytest.raises(ValueError, match="num_classes must be 3"):
            SyntheticMRIDataset(num_samples=10, num_classes=4)


class TestMRI25DDataset:

    def test_load_from_dataframe(self, tmp_path):
        # Create temporary tensor files
        tensor_dir = tmp_path / "tensors"
        tensor_dir.mkdir()

        records = []
        for i, diag in enumerate(["CN", "MCI", "AD"]):
            arr = np.ones((3, 224, 224), dtype=np.float32) * (i + 1)
            t_path = tensor_dir / f"sample_{i}.npy"
            np.save(t_path, arr)
            records.append({
                "Subject_ID": f"OAS1_{i:04d}",
                "Session_ID": "MR1",
                "Cohort_Source": "OASIS",
                "Diagnosis_Class": diag,
                "Diagnosis_Numeric": i,
                "Output_Tensor_Path": str(t_path),
            })

        df = pd.DataFrame(records)
        ds = MRI25DDataset(manifest=df)
        assert len(ds) == 3

        item0 = ds[0]
        assert item0["image"].shape == (3, 224, 224)
        assert item0["label"].item() == 0
        assert item0["diagnosis_name"] == "CN"
        assert item0["subject_id"] == "OAS1_0000"

    def test_missing_file_raises_error(self, tmp_path):
        df = pd.DataFrame([{
            "Subject_ID": "SUBJ_01",
            "Diagnosis_Class": "CN",
            "Output_Tensor_Path": str(tmp_path / "non_existent.npy"),
        }])
        ds = MRI25DDataset(manifest=df)
        with pytest.raises(FileNotFoundError):
            _ = ds[0]

    def test_invalid_tensor_shape_raises(self, tmp_path):
        bad_tensor_path = tmp_path / "bad.npy"
        np.save(bad_tensor_path, np.ones((1, 100, 100), dtype=np.float32))

        df = pd.DataFrame([{
            "Subject_ID": "SUBJ_01",
            "Diagnosis_Numeric": 0,
            "Output_Tensor_Path": str(bad_tensor_path),
        }])
        ds = MRI25DDataset(manifest=df)
        with pytest.raises(ValueError, match="Invalid tensor shape"):
            _ = ds[0]

    def test_non_finite_tensor_raises(self, tmp_path):
        nan_tensor_path = tmp_path / "nan.npy"
        arr = np.ones((3, 224, 224), dtype=np.float32)
        arr[0, 0, 0] = np.nan
        np.save(nan_tensor_path, arr)

        df = pd.DataFrame([{
            "Subject_ID": "SUBJ_01",
            "Diagnosis_Numeric": 1,
            "Output_Tensor_Path": str(nan_tensor_path),
        }])
        ds = MRI25DDataset(manifest=df)
        with pytest.raises(ValueError, match="non-finite"):
            _ = ds[0]


class TestMRIDataLoader:

    def test_collate_fn(self):
        batch = [
            {
                "image": torch.ones(3, 224, 224, dtype=torch.float32),
                "label": torch.tensor(0, dtype=torch.int64),
                "subject_id": "S1",
                "session_id": "SES1",
                "cohort_source": "C1",
                "diagnosis_name": "CN",
            },
            {
                "image": torch.ones(3, 224, 224, dtype=torch.float32) * 2,
                "label": torch.tensor(1, dtype=torch.int64),
                "subject_id": "S2",
                "session_id": "SES1",
                "cohort_source": "C1",
                "diagnosis_name": "MCI",
            },
        ]
        collated = mri_collate_fn(batch)
        assert collated["images"].shape == (2, 3, 224, 224)
        assert collated["labels"].shape == (2,)
        assert collated["subject_ids"] == ["S1", "S2"]

    def test_create_dataloader(self):
        ds = SyntheticMRIDataset(num_samples=10, seed=42)
        loader = create_mri_dataloader(ds, batch_size=4, shuffle=False)
        assert len(loader) == 3  # 4 + 4 + 2

        batches = list(loader)
        assert batches[0]["images"].shape == (4, 3, 224, 224)
        assert batches[1]["images"].shape == (4, 3, 224, 224)
        assert batches[2]["images"].shape == (2, 3, 224, 224)
