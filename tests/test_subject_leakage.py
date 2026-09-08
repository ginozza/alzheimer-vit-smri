"""
Strict Subject-Level Leakage Prevention Tests.
Direct evidence for Deliverable E1 Acceptance Criterion:
"Ningún sujeto aparece en más de una partición (Train, Validation, Test)."
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data.inventory_loader import DatasetInventoryLoader
from src.data.subject_split import SubjectSplitter


@pytest.fixture
def sample_dataset():
    loader = DatasetInventoryLoader()
    paths = [
        Path("data/inventory/adni_metadata_inventory.csv"),
        Path("data/inventory/oasis_metadata_inventory.csv"),
    ]
    combined_df, _ = loader.load_combined_cohort(paths)
    return combined_df


def test_zero_subject_leakage_holdout(sample_dataset):
    """
    Formally verifies:
    S_train ∩ S_val = ∅
    S_train ∩ S_test = ∅
    S_val ∩ S_test = ∅
    """
    splitter = SubjectSplitter(random_seed=42)
    result = splitter.split_holdout(sample_dataset, test_size=0.20, val_size_from_dev=0.125)

    s_train = set(result.train_df["Subject_ID"])
    s_val = set(result.val_df["Subject_ID"])
    s_test = set(result.test_df["Subject_ID"])

    # 1. Zero leakage assertions
    assert len(s_train.intersection(s_val)) == 0, "Train and Val must share 0 subjects"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test must share 0 subjects"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test must share 0 subjects"

    # 2. Complete partition assertion: All subjects accounted for
    all_partitioned_subjects = s_train.union(s_val).union(s_test)
    assert all_partitioned_subjects == set(sample_dataset["Subject_ID"])

    # 3. Longitudinal coherence assertion:
    # If a subject has multiple scans/sessions, all must be in the same partition!
    multi_session_subjects = (
        sample_dataset.groupby("Subject_ID").filter(lambda x: len(x) > 1)["Subject_ID"].unique()
    )
    for subj in multi_session_subjects:
        in_train = subj in s_train
        in_val = subj in s_val
        in_test = subj in s_test
        # Sum of truth values must be exactly 1
        assert sum([in_train, in_val, in_test]) == 1, (
            f"Multi-session subject {subj} leaked into multiple partitions!"
        )


def test_zero_leakage_kfold_cv(sample_dataset):
    """
    Verifies that within the development pool (80%), Stratified 5-Fold Cross Validation
    maintains zero subject overlap between each fold's train and validation subsets.
    """
    splitter = SubjectSplitter(random_seed=42)
    result = splitter.split_holdout(sample_dataset, test_size=0.20)
    dev_df = result.development_df

    folds = splitter.generate_kfold_cv(dev_df, n_splits=5)
    assert len(folds) == 5

    for fold_idx, (f_train, f_val) in enumerate(folds):
        s_train = set(f_train["Subject_ID"])
        s_val = set(f_val["Subject_ID"])
        overlap = s_train.intersection(s_val)
        assert len(overlap) == 0, f"Fold {fold_idx} has subject leakage: {overlap}"


def test_leakage_detector_raises_on_contamination():
    """
    Tests that the verify_zero_leakage static checker correctly detects and rejects
    any accidental data contamination or subject overlap.
    """
    splitter = SubjectSplitter()

    # Create artificial contaminated DataFrames
    train_df = pd.DataFrame({"Subject_ID": ["SUBJ_1", "SUBJ_2", "SUBJ_3"]})
    val_df = pd.DataFrame({"Subject_ID": ["SUBJ_4", "SUBJ_5"]})
    # Contaminate test with SUBJ_1
    contaminated_test_df = pd.DataFrame({"Subject_ID": ["SUBJ_1", "SUBJ_6"]})

    with pytest.raises(AssertionError, match="Subject leakage detected"):
        splitter.verify_zero_leakage(train_df, val_df, contaminated_test_df)
