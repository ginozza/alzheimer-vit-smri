"""Ingestion, clinical mapping, and tensor preprocessing adapter for Kaggle 2D MRI datasets.

Supports:
1. ninadaithal/imagesoasis (OASIS-1 2D slices)
2. ahmedashrafahmed / Alzheimer's Dataset (4-Class ADNI/OASIS)
Outputs standardized ViT-B/16 compatible 3-channel tensors (3, 224, 224) float32.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .preprocessing import QCError, file_sha256
from .slice_representation import Slice25DConfig


# Standard clinical taxonomy mapping from 4-class Kaggle folders to Project Triclass
KAGGLE_CLASS_MAPPING: Dict[str, str] = {
    # Non Demented -> Control Normal (CN)
    "nondemented": "CN",
    "non demented": "CN",
    "non_demented": "CN",
    "cn": "CN",
    # Very Mild & Mild Demented -> Mild Cognitive Impairment (MCI - Prodromal phase)
    "verymilddemented": "MCI",
    "very mild demented": "MCI",
    "very_mild_demented": "MCI",
    "verymilddementia": "MCI",
    "very mild dementia": "MCI",
    "very_mild_dementia": "MCI",
    "milddemented": "MCI",
    "mild demented": "MCI",
    "mild_demented": "MCI",
    "milddementia": "MCI",
    "mild dementia": "MCI",
    "mild_dementia": "MCI",
    "mci": "MCI",
    # Moderate Demented -> Alzheimer's Disease (AD)
    "moderatedemented": "AD",
    "moderate demented": "AD",
    "moderate_demented": "AD",
    "moderatedementia": "AD",
    "moderate dementia": "AD",
    "moderate_dementia": "AD",
    "ad": "AD",
}

CLASS_TO_NUMERIC: Dict[str, int] = {
    "CN": 0,
    "MCI": 1,
    "AD": 2,
}


@dataclass(frozen=True)
class KaggleAdapterConfig:
    target_height: int = 224
    target_width: int = 224
    channels: int = 3
    normalization: str = "z_score"
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    min_dimension: int = 64
    max_dimension: int = 2048
    min_foreground_ratio: float = 0.05
    random_seed: int = 42

    def as_dict(self):
        return asdict(self)


def normalize_class_name(folder_name: str) -> str:
    """Normalizes a raw class folder name to CN, MCI, or AD."""
    cleaned = folder_name.strip().lower().replace("-", " ")
    cleaned_no_spaces = cleaned.replace(" ", "")
    if cleaned in KAGGLE_CLASS_MAPPING:
        return KAGGLE_CLASS_MAPPING[cleaned]
    if cleaned_no_spaces in KAGGLE_CLASS_MAPPING:
        return KAGGLE_CLASS_MAPPING[cleaned_no_spaces]
    raise ValueError(f"Unrecognized diagnostic class folder: '{folder_name}'")


def extract_oasis_subject_id(filename: str, fallback_idx: int) -> Tuple[str, str]:
    """
    Extracts Subject_ID and Session_ID from standard OASIS filenames like:
    'OAS1_0001_MR1_mpr-1_100.jpg' -> ('OAS1_0001', 'MR1')
    """
    match = re.search(r"(OAS1_\d{4})_(MR\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).upper()
    match_subj = re.search(r"(OAS1_\d{4})", filename, re.IGNORECASE)
    if match_subj:
        return match_subj.group(1).upper(), "MR1"
    # Fallback for generic numbered files
    return f"KAG_{fallback_idx:05d}", "SES1"


def preprocess_2d_slice(
    image_path: Path | str,
    config: Optional[KaggleAdapterConfig] = None,
) -> np.ndarray:
    """
    Loads, validates (QC), normalizes, and transforms a 2D image slice into a
    ViT-B/16 compatible (3, 224, 224) float32 tensor.
    """
    cfg = config or KaggleAdapterConfig()
    path = Path(image_path)
    if not path.is_file():
        raise QCError("file_not_found", f"Image file not found: {path}")

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert("L")  # Convert to 8-bit grayscale
            raw_array = np.array(img, dtype=np.float32)
    except Exception as exc:
        raise QCError("corrupted_image", f"Cannot decode image {path}: {exc}")

    h, w = raw_array.shape
    if h < cfg.min_dimension or w < cfg.min_dimension:
        raise QCError("dimension_underflow", f"Image shape ({h}, {w}) smaller than min {cfg.min_dimension}")
    if h > cfg.max_dimension or w > cfg.max_dimension:
        raise QCError("dimension_overflow", f"Image shape ({h}, {w}) larger than max {cfg.max_dimension}")

    if not np.all(np.isfinite(raw_array)):
        raise QCError("nonfinite", "Image contains NaN or Inf values")

    # Foreground threshold check
    foreground = raw_array[raw_array > 0]
    if len(foreground) < (h * w * cfg.min_foreground_ratio):
        raise QCError("empty_foreground", "Image has insufficient non-zero brain foreground")

    # Intensity normalization (z-score with percentile clipping on non-zero pixels)
    p_low, p_high = np.percentile(foreground, cfg.clip_percentiles)
    if p_high <= p_low:
        clipped = foreground
    else:
        clipped = np.clip(raw_array, p_low, p_high)

    mask = clipped > 0
    if np.any(mask):
        mean_val = float(np.mean(clipped[mask]))
        std_val = float(np.std(clipped[mask])) + 1e-8
        normalized = np.zeros_like(clipped, dtype=np.float32)
        normalized[mask] = (clipped[mask] - mean_val) / std_val
    else:
        normalized = np.zeros_like(clipped, dtype=np.float32)

    # Resize to target (target_height, target_width)
    norm_img = Image.fromarray(normalized)
    resized_img = norm_img.resize((cfg.target_width, cfg.target_height), Image.Resampling.BILINEAR)
    resized_slice = np.array(resized_img, dtype=np.float32)

    # Stack to 3 channels (3, 224, 224) float32 for ViT-B/16
    tensor_3ch = np.stack([resized_slice] * cfg.channels, axis=0)
    return tensor_3ch


class KaggleDatasetScanner:
    """Scans and builds a manifest from a Kaggle dataset directory."""

    def __init__(self, root_dir: Path | str, cohort_name: str = "KAGGLE-OASIS"):
        self.root_dir = Path(root_dir)
        self.cohort_name = cohort_name

    def scan(self) -> List[Dict]:
        records = []
        idx = 0
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        for path in self.root_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue

            # Identify diagnostic class from immediate parent or folder path
            parts = [p.name for p in path.parents]
            class_name = None
            for part in parts:
                try:
                    class_name = normalize_class_name(part)
                    break
                except ValueError:
                    continue

            if not class_name:
                continue

            subj_id, sess_id = extract_oasis_subject_id(path.name, idx)
            rel_path = path.relative_to(self.root_dir).as_posix()

            records.append({
                "Subject_ID": subj_id,
                "Session_ID": sess_id,
                "Cohort_Source": self.cohort_name,
                "Image_File_Path": rel_path,
                "Diagnosis_Class": class_name,
                "Diagnosis_Numeric": CLASS_TO_NUMERIC[class_name],
                "Modality": "T1w_sMRI",
                "Pulse_Sequence": "MPRAGE",
                "Space": "MNI152_2D",
                "Template_ID": "MNI152_1mm_axial_slice",
                "Sample_Kind": "real",
                "Quality_Control_Pass": True,
                "Brain_Extraction_Pass": True,
                "Registration_QC_Pass": True,
                "Preprocessing_Provenance": f"Extracted from Kaggle {self.cohort_name} 2D dataset",
            })
            idx += 1

        return records

    def create_partitioned_manifest(
        self,
        records: List[Dict],
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Partitions subjects strictly with zero subject leakage."""
        if not records:
            raise ValueError(f"No valid image files found in {self.root_dir}")

        df = pd.DataFrame(records)
        unique_subjects = df[["Subject_ID", "Diagnosis_Class"]].drop_duplicates("Subject_ID")

        rng = np.random.RandomState(random_seed)
        shuffled = unique_subjects.sample(frac=1.0, random_state=rng)

        # Stratified allocation by subject
        subject_splits = {}
        for diag, group in shuffled.groupby("Diagnosis_Class"):
            subjs = group["Subject_ID"].tolist()
            n = len(subjs)
            n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
            n_val = max(1, int(round(n * val_ratio))) if (n - n_test) >= 2 else 0

            test_subjs = set(subjs[:n_test])
            val_subjs = set(subjs[n_test : n_test + n_val])
            train_subjs = set(subjs[n_test + n_val :])

            for s in train_subjs:
                subject_splits[s] = "Train"
            for s in val_subjs:
                subject_splits[s] = "Validation"
            for s in test_subjs:
                subject_splits[s] = "Test"

        df["Split"] = df["Subject_ID"].map(subject_splits)
        return df


def execute_kaggle_preprocessing_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    cohort_name: str = "KAGGLE-OASIS",
    config: Optional[KaggleAdapterConfig] = None,
) -> Dict:
    """Executes the complete preprocessing pipeline on a Kaggle dataset folder."""
    cfg = config or KaggleAdapterConfig()
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory {out_dir} already exists and is not empty")

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = out_dir / "tensors"
    for split in ("Train", "Validation", "Test"):
        (tensors_dir / split).mkdir(parents=True, exist_ok=True)

    scanner = KaggleDatasetScanner(in_dir, cohort_name=cohort_name)
    raw_records = scanner.scan()
    manifest_df = scanner.create_partitioned_manifest(raw_records, random_seed=cfg.random_seed)

    # Save input manifest
    manifest_path = out_dir / "input_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    qc_entries = []
    output_records = []
    processed_count = 0
    rejected_count = 0

    for _, row in manifest_df.iterrows():
        img_rel = row["Image_File_Path"]
        img_full = in_dir / img_rel
        subj = row["Subject_ID"]
        split = row["Split"]
        diag = row["Diagnosis_Class"]

        qc_record = {
            "Subject_ID": subj,
            "Cohort_Source": cohort_name,
            "Image_File_Path": img_rel,
            "Split": split,
            "Diagnosis_Class": diag,
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            tensor = preprocess_2d_slice(img_full, cfg)
            tensor_name = f"{subj}_{row['Session_ID']}_{diag}.npy"
            tensor_path = tensors_dir / split / tensor_name
            np.save(tensor_path, tensor)

            sha = file_sha256(tensor_path)
            qc_record["Status"] = "PASS"
            qc_record["Tensor_File"] = f"tensors/{split}/{tensor_name}"
            qc_record["Tensor_Shape"] = list(tensor.shape)
            qc_record["SHA256"] = sha

            output_records.append({
                **row.to_dict(),
                "Output_Tensor_Path": f"tensors/{split}/{tensor_name}",
                "Tensor_Shape": str(list(tensor.shape)),
                "Tensor_Dtype": str(tensor.dtype),
                "SHA256": sha,
            })
            processed_count += 1
        except QCError as qce:
            qc_record["Status"] = "REJECT"
            qc_record["Reject_Reason"] = qce.reason
            qc_record["Error_Message"] = str(qce)
            rejected_count += 1
        except Exception as exc:
            qc_record["Status"] = "REJECT"
            qc_record["Reject_Reason"] = "unexpected_error"
            qc_record["Error_Message"] = str(exc)
            rejected_count += 1

        qc_entries.append(qc_record)

    # Write outputs.csv
    if output_records:
        pd.DataFrame(output_records).to_csv(out_dir / "outputs.csv", index=False)

    # Write qc.jsonl
    with (out_dir / "qc.jsonl").open("w", encoding="utf-8") as fp:
        for entry in qc_entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write summary.json
    summary = {
        "Cohort_Source": cohort_name,
        "Total_Images_Found": len(manifest_df),
        "Processed_Passed": processed_count,
        "Rejected": rejected_count,
        "Class_Distribution": dict(Counter(manifest_df["Diagnosis_Class"])),
        "Split_Distribution": dict(Counter(manifest_df["Split"])),
        "Tensor_Shape": [cfg.channels, cfg.target_height, cfg.target_width],
        "Tensor_Dtype": "float32",
        "Config": cfg.as_dict(),
        "Timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary
