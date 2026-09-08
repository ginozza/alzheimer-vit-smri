"""
Test suite for metadata inventory loading and inclusion/exclusion criteria.
Verifies acceptance criteria for E1: Permissions, sources, classes and exclusions defined.
"""

from pathlib import Path
import pytest
from src.data.inventory_loader import DatasetInventoryLoader


@pytest.fixture
def inventory_loader():
    dict_path = Path("data/inventory/adni_oasis_data_dictionary.yaml")
    return DatasetInventoryLoader(dictionary_path=dict_path)


def test_adni_inventory_loading(inventory_loader):
    path = Path("data/inventory/adni_metadata_inventory.csv")
    assert path.exists(), "ADNI inventory file must exist"

    df = inventory_loader.load_inventory(path)
    assert not df.empty, "ADNI inventory must not be empty"
    assert "Subject_ID" in df.columns
    assert "Diagnosis_Class" in df.columns
    assert "Quality_Control_Pass" in df.columns

    # Classes must be in {CN, MCI, AD}
    unique_classes = set(df["Diagnosis_Class"].unique())
    assert unique_classes.issubset({"CN", "MCI", "AD"})


def test_oasis_inventory_loading(inventory_loader):
    path = Path("data/inventory/oasis_metadata_inventory.csv")
    assert path.exists(), "OASIS inventory file must exist"

    df = inventory_loader.load_inventory(path)
    assert not df.empty, "OASIS inventory must not be empty"
    assert "Subject_ID" in df.columns
    assert "Diagnosis_Class" in df.columns


def test_inclusion_exclusion_filtering(inventory_loader):
    path = Path("data/inventory/adni_metadata_inventory.csv")
    df = inventory_loader.load_inventory(path)

    # Some test records have Quality_Control_Pass=false or Age < 55
    filtered_df, audit = inventory_loader.apply_criteria(df, min_age=55.0, require_qc=True)

    assert audit["excluded_qc_failure"] >= 1, "Should identify and exclude QC failures"
    assert (filtered_df["Age"] >= 55.0).all(), "All included subjects must be >= 55 years old"
    assert (filtered_df["Quality_Control_Pass"] == True).all(), "All included scans must pass QC"


def test_combined_cohort_aggregation(inventory_loader):
    paths = [
        Path("data/inventory/adni_metadata_inventory.csv"),
        Path("data/inventory/oasis_metadata_inventory.csv"),
    ]
    combined_df, report = inventory_loader.load_combined_cohort(paths)

    assert not combined_df.empty
    assert "combined" in report
    assert report["combined"]["total_scans"] == len(combined_df)
    assert set(report["combined"]["class_distribution"].keys()).issubset({"CN", "MCI", "AD"})
