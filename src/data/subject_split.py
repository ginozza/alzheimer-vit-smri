"""
Subject-level data partitioning module with zero data-leakage guarantee.
Seminario III - Periodo 2026-II, Universidad del Magdalena.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


@dataclass
class DataSplitResult:
    """Holds partitioned DataFrames and split metadata."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    development_df: pd.DataFrame
    fold_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None
    audit_summary: Optional[dict] = None


class SubjectSplitter:
    """
    Executes subject-level stratified partitioning to completely eliminate data leakage.
    Ensures that all slices/scans from any given Subject_ID belong strictly to exactly
    one subset (Train, Validation, or Test) and never cross partition boundaries.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def split_holdout(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        val_size_from_dev: float = 0.125,  # 0.125 * 0.80 = 0.10 of total (70% train, 10% val, 20% test)
        subject_col: str = "Subject_ID",
        class_col: str = "Diagnosis_Class",
    ) -> DataSplitResult:
        """
        Splits dataset by unique Subject_ID using stratified sampling.

        1. Aggregates to unique subjects and their majority/baseline diagnosis.
        2. Splits subjects into Development (80%) and Hold-out Test (20%).
        3. Optionally splits Development subjects into Train (70% total) and Val (10% total).
        4. Maps back to full scan/session table.
        5. Verifies zero subject intersection.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Step 1: Extract unique subject profiles for stratification
        # If subject has multiple visits, we use their primary/initial diagnosis
        subject_profiles = (
            df.groupby(subject_col)
            .agg(
                {
                    class_col: "first",
                    "Sex": "first",
                    "Age": "mean",
                }
            )
            .reset_index()
        )

        subjects = subject_profiles[subject_col].values
        labels = subject_profiles[class_col].values

        # Step 2: Stratified split for 80% Dev / 20% Test
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_seed
        )
        dev_subj_idx, test_subj_idx = next(sss_test.split(subjects, labels))

        dev_subjects = set(subjects[dev_subj_idx])
        test_subjects = set(subjects[test_subj_idx])

        # Step 3: Split Development into Train and Val
        dev_profiles = subject_profiles.iloc[dev_subj_idx].reset_index(drop=True)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size_from_dev, random_state=self.random_seed
        )
        train_subj_idx, val_subj_idx = next(
            sss_val.split(
                dev_profiles[subject_col].values, dev_profiles[class_col].values
            )
        )

        train_subjects = set(dev_profiles[subject_col].values[train_subj_idx])
        val_subjects = set(dev_profiles[subject_col].values[val_subj_idx])

        # Step 4: Map back to all scans
        train_df = df[df[subject_col].isin(train_subjects)].copy().reset_index(drop=True)
        val_df = df[df[subject_col].isin(val_subjects)].copy().reset_index(drop=True)
        test_df = df[df[subject_col].isin(test_subjects)].copy().reset_index(drop=True)
        dev_df = df[df[subject_col].isin(dev_subjects)].copy().reset_index(drop=True)

        train_df["Split"] = "Train"
        val_df["Split"] = "Validation"
        test_df["Split"] = "Test"
        dev_df["Split"] = "Development"

        result = DataSplitResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            development_df=dev_df,
        )

        # Step 5: Enforce zero-leakage verification
        leakage_check = self.verify_zero_leakage(train_df, val_df, test_df, subject_col)
        result.audit_summary = {
            "random_seed": self.random_seed,
            "leakage_verified": leakage_check["zero_leakage_confirmed"],
            "subject_counts": {
                "train_subjects": len(train_subjects),
                "val_subjects": len(val_subjects),
                "test_subjects": len(test_subjects),
                "total_unique": len(set(subjects)),
            },
            "scan_counts": {
                "train_scans": len(train_df),
                "val_scans": len(val_df),
                "test_scans": len(test_df),
                "total_scans": len(df),
            },
            "class_distribution": {
                "train": train_df[class_col].value_counts().to_dict(),
                "val": val_df[class_col].value_counts().to_dict(),
                "test": test_df[class_col].value_counts().to_dict(),
            },
        }

        return result

    def generate_kfold_cv(
        self,
        dev_df: pd.DataFrame,
        n_splits: int = 5,
        subject_col: str = "Subject_ID",
        class_col: str = "Diagnosis_Class",
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates stratified K-Fold cross validation on the development pool,
        strictly grouping by Subject_ID so no subject is in both train and validation folds.
        """
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        folds = []

        X = dev_df.index.values
        y = dev_df[class_col].values
        groups = dev_df[subject_col].values

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            fold_train = dev_df.iloc[train_idx].copy().reset_index(drop=True)
            fold_val = dev_df.iloc[val_idx].copy().reset_index(drop=True)

            # Assert zero leakage between fold train and fold val
            train_subjs = set(fold_train[subject_col])
            val_subjs = set(fold_val[subject_col])
            overlap = train_subjs.intersection(val_subjs)
            if overlap:
                raise ValueError(
                    f"Fold {fold_idx} detected subject leakage: {len(overlap)} subjects overlap."
                )

            fold_train["Fold"] = fold_idx
            fold_val["Fold"] = fold_idx
            folds.append((fold_train, fold_val))

        return folds

    @staticmethod
    def verify_zero_leakage(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        subject_col: str = "Subject_ID",
    ) -> dict:
        """
        Formally proves whether subject leakage is 0.
        Mathematical criterion: S_train ∩ S_val = ∅, S_train ∩ S_test = ∅, S_val ∩ S_test = ∅
        """
        s_train = set(train_df[subject_col].unique())
        s_val = set(val_df[subject_col].unique())
        s_test = set(test_df[subject_col].unique())

        train_val_overlap = s_train.intersection(s_val)
        train_test_overlap = s_train.intersection(s_test)
        val_test_overlap = s_val.intersection(s_test)

        is_zero_leakage = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        report = {
            "zero_leakage_confirmed": is_zero_leakage,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap_ids": list(train_val_overlap),
            "train_test_overlap_ids": list(train_test_overlap),
            "val_test_overlap_ids": list(val_test_overlap),
        }

        if not is_zero_leakage:
            raise AssertionError(f"Subject leakage detected! Audit report: {report}")

        return report
