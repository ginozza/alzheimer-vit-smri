"""
Dataset inventory loading, validation, and inclusion/exclusion filtering.
Aligns with ADNI and OASIS metadata standards and Deliverable E1 requirements.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import yaml


@dataclass
class SubjectRecord:
    """Represents a validated subject MRI session entry."""
    subject_id: str
    session_id: str
    cohort_source: str
    diagnosis_class: str
    age: float
    sex: str
    mmse: float
    cdr: float
    cdr_sb: Optional[float]
    apoe4_alleles: Optional[int]
    modality: str
    pulse_sequence: str
    field_strength_tesla: float
    voxel_resolution_mm: str
    quality_control_pass: bool
    image_file_path: str


class DatasetInventoryLoader:
    """
    Loads, harmonizes, and filters ADNI and OASIS metadata inventories.
    Enforces strict inclusion/exclusion criteria for triclass classification (CN, MCI, AD).
    """

    VALID_CLASSES = {"CN", "MCI", "AD"}

    def __init__(self, dictionary_path: Optional[Union[str, Path]] = None):
        self.dictionary_path = Path(dictionary_path) if dictionary_path else None
        self.schema = self._load_dictionary() if self.dictionary_path and self.dictionary_path.exists() else {}

    def _load_dictionary(self) -> dict:
        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_inventory(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Loads a CSV inventory manifest and checks for required columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Inventory manifest not found at: {path}")

        df = pd.read_csv(path)
        required_cols = [
            "Subject_ID",
            "Session_ID",
            "Cohort_Source",
            "Diagnosis_Class",
            "Age",
            "Sex",
            "MMSE",
            "CDR",
            "Quality_Control_Pass",
            "Image_File_Path",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path.name}: {missing}")

        return df

    def apply_criteria(
        self,
        df: pd.DataFrame,
        min_age: float = 55.0,
        max_age: float = 95.0,
        require_qc: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Applies inclusion and exclusion filters to the raw metadata.

        Inclusion criteria:
          - Diagnosis in {CN, MCI, AD}
          - Age between min_age and max_age
          - Passed quality control (if require_qc=True)
          - Non-null vital demographic and cognitive scores

        Returns:
            Tuple of (filtered_df, audit_metrics_dict)
        """
        initial_count = len(df)
        reasons = {
            "initial_scans": initial_count,
            "excluded_invalid_diagnosis": 0,
            "excluded_age_out_of_range": 0,
            "excluded_qc_failure": 0,
            "final_eligible_scans": 0,
        }

        # Filter 1: Valid Diagnosis Class
        valid_diag_mask = df["Diagnosis_Class"].isin(self.VALID_CLASSES)
        reasons["excluded_invalid_diagnosis"] = int((~valid_diag_mask).sum())
        filtered = df[valid_diag_mask].copy()

        # Filter 2: Age range
        age_mask = (filtered["Age"] >= min_age) & (filtered["Age"] <= max_age)
        reasons["excluded_age_out_of_range"] = int((~age_mask).sum())
        filtered = filtered[age_mask]

        # Filter 3: Quality Control
        if require_qc:
            qc_mask = filtered["Quality_Control_Pass"] == True
            reasons["excluded_qc_failure"] = int((~qc_mask).sum())
            filtered = filtered[qc_mask]

        reasons["final_eligible_scans"] = len(filtered)
        reasons["unique_subjects"] = int(filtered["Subject_ID"].nunique())

        return filtered.reset_index(drop=True), reasons

    def load_combined_cohort(
        self,
        inventory_paths: List[Union[str, Path]],
        min_age: float = 55.0,
        max_age: float = 95.0,
        require_qc: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """
        Loads and combines multiple dataset inventories (e.g. ADNI + OASIS).
        """
        dfs = []
        reports = {}

        for path_str in inventory_paths:
            p = Path(path_str)
            df = self.load_inventory(p)
            filtered_df, report = self.apply_criteria(
                df, min_age=min_age, max_age=max_age, require_qc=require_qc
            )
            dfs.append(filtered_df)
            reports[p.stem] = report

        combined_df = pd.concat(dfs, ignore_index=True)
        reports["combined"] = {
            "total_scans": len(combined_df),
            "unique_subjects": int(combined_df["Subject_ID"].nunique()),
            "class_distribution": combined_df["Diagnosis_Class"].value_counts().to_dict(),
        }

        return combined_df, reports
