"""PyTorch Dataset implementations for 2.5D MRI tensors and synthetic verification.

Supports loading preprocessed tensors (3, 224, 224) float32 from E2 output manifests,
direct numpy files, or synthetically generated reproducible cohorts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES: Tuple[str, ...] = ("CN", "MCI", "AD")
CLASS_TO_IDX: Dict[str, int] = {"CN": 0, "MCI": 1, "AD": 2}
IDX_TO_CLASS: Dict[int, str] = {0: "CN", 1: "MCI", 2: "AD"}


@dataclass(frozen=True)
class MRISample:
    """Represents a single loaded MRI slice sample with clinical metadata."""

    image: torch.Tensor
    label: torch.Tensor
    subject_id: str
    session_id: str
    cohort_source: str
    diagnosis_name: str

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "label": self.label,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "cohort_source": self.cohort_source,
            "diagnosis_name": self.diagnosis_name,
        }


class MRI25DDataset(Dataset):
    """Dataset for preprocessed 2.5D sMRI axial tensors.

    Accepts:
    1. A pandas DataFrame or path to manifest CSV (outputs.csv or input_manifest.csv)
       containing columns: Subject_ID, Diagnosis_Numeric (or Diagnosis_Class), and
       Output_Tensor_Path (or Image_File_Path).
    2. An optional base directory where relative tensor paths are located.
    3. An optional transform callable applied to the (3, 224, 224) torch.Tensor.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy().reset_index(drop=True)
        else:
            raise TypeError(f"manifest must be a DataFrame, str or Path, got {type(manifest)}")

        if len(self.df) == 0:
            raise ValueError("Manifest contains zero records")

        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.transform = transform
        self._validate_manifest_columns()

    def _validate_manifest_columns(self) -> None:
        """Ensure required metadata and path columns exist."""
        # Check label column
        if "Diagnosis_Numeric" not in self.df.columns and "Diagnosis_Class" not in self.df.columns:
            raise ValueError("Manifest must contain 'Diagnosis_Numeric' or 'Diagnosis_Class' column")

        # Check path column
        path_cols = ["Output_Tensor_Path", "Image_File_Path", "tensor_path", "file_path"]
        found_path_col = None
        for col in path_cols:
            if col in self.df.columns:
                found_path_col = col
                break
        if found_path_col is None:
            raise ValueError(f"Manifest must contain a path column from: {path_cols}")
        self.path_column = found_path_col

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        if self.base_dir is not None:
            return self.base_dir / path
        return path

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        file_path = self._resolve_path(str(row[self.path_column]))

        if not file_path.is_file():
            raise FileNotFoundError(f"Tensor file not found at: {file_path}")

        try:
            arr = np.load(file_path)
        except Exception as exc:
            raise IOError(f"Failed to load tensor from {file_path}: {exc}") from exc

        tensor = torch.from_numpy(arr)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        # Validate dimensional integrity (C, H, W) == (3, 224, 224)
        if tensor.ndim != 3 or tensor.shape != (3, 224, 224):
            raise ValueError(
                f"Invalid tensor shape {tuple(tensor.shape)} at index {idx}. "
                f"Expected (3, 224, 224)"
            )

        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"Tensor at index {idx} contains non-finite values (NaN/Inf)")

        if self.transform is not None:
            tensor = self.transform(tensor)

        # Resolve class label
        if "Diagnosis_Numeric" in row and pd.notna(row["Diagnosis_Numeric"]):
            label_idx = int(row["Diagnosis_Numeric"])
        else:
            class_str = str(row["Diagnosis_Class"]).strip()
            if class_str not in CLASS_TO_IDX:
                raise ValueError(f"Unknown diagnostic class '{class_str}' at index {idx}")
            label_idx = CLASS_TO_IDX[class_str]

        if label_idx not in (0, 1, 2):
            raise ValueError(f"Invalid label index {label_idx} at index {idx}. Must be 0, 1, or 2")

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        subject_id = str(row.get("Subject_ID", f"SUBJ_{idx:04d}"))
        session_id = str(row.get("Session_ID", "SES01"))
        cohort = str(row.get("Cohort_Source", "UNKNOWN"))

        return {
            "image": tensor,
            "label": label_tensor,
            "subject_id": subject_id,
            "session_id": session_id,
            "cohort_source": cohort,
            "diagnosis_name": IDX_TO_CLASS[label_idx],
        }


class SyntheticMRIDataset(Dataset):
    """Generates synthetic, deterministic MRI tensor batches for testing and pipeline audits.

    Does not require external disk files, guaranteeing zero I/O dependency in CI/CD.
    """

    def __init__(
        self,
        num_samples: int = 24,
        num_classes: int = 3,
        image_shape: Tuple[int, int, int] = (3, 224, 224),
        seed: int = 42,
    ):
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_classes != 3:
            raise ValueError("num_classes must be 3 for triclass Alzheimer classification")

        self.num_samples = num_samples
        self.image_shape = image_shape
        rng = np.random.RandomState(seed)

        # Balanced synthetic distribution
        self.labels = [i % num_classes for i in range(num_samples)]
        rng.shuffle(self.labels)

        # Pre-generate noise tensors with distinct class signatures
        self.data: List[torch.Tensor] = []
        for i, lbl in enumerate(self.labels):
            # Base Gaussian noise
            arr = rng.randn(*image_shape).astype(np.float32) * 0.5
            # Add slight signal offset per class
            arr += (lbl * 0.2)
            self.data.append(torch.from_numpy(arr))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        lbl = self.labels[idx]
        return {
            "image": self.data[idx],
            "label": torch.tensor(lbl, dtype=torch.long),
            "subject_id": f"SYNTH_{idx:04d}",
            "session_id": "MR01",
            "cohort_source": "SYNTHETIC-AUDIT",
            "diagnosis_name": IDX_TO_CLASS[lbl],
        }
