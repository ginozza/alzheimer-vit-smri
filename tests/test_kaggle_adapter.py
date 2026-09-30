"""Unit tests for the Kaggle 2D MRI dataset adapter and preprocessing pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data.kaggle_adapter import (
    KaggleAdapterConfig,
    KaggleDatasetScanner,
    extract_oasis_subject_id,
    normalize_class_name,
    preprocess_2d_slice,
    execute_kaggle_preprocessing_pipeline,
)
from src.data.preprocessing import QCError


def test_normalize_class_name():
    # Non Demented -> CN
    assert normalize_class_name("Non Demented") == "CN"
    assert normalize_class_name("NonDemented") == "CN"
    assert normalize_class_name("non_demented") == "CN"
    assert normalize_class_name("cn") == "CN"

    # Very Mild & Mild -> MCI (both 'Demented' and 'Dementia' variants)
    assert normalize_class_name("Very Mild Demented") == "MCI"
    assert normalize_class_name("VeryMildDemented") == "MCI"
    assert normalize_class_name("Very mild Dementia") == "MCI"
    assert normalize_class_name("Mild Demented") == "MCI"
    assert normalize_class_name("MildDemented") == "MCI"
    assert normalize_class_name("Mild Dementia") == "MCI"
    assert normalize_class_name("mci") == "MCI"

    # Moderate -> AD (both 'Demented' and 'Dementia' variants)
    assert normalize_class_name("Moderate Demented") == "AD"
    assert normalize_class_name("ModerateDemented") == "AD"
    assert normalize_class_name("Moderate Dementia") == "AD"
    assert normalize_class_name("moderate_demented") == "AD"
    assert normalize_class_name("ad") == "AD"

    with pytest.raises(ValueError, match="Unrecognized"):
        normalize_class_name("UnknownClass")


def test_extract_oasis_subject_id():
    subj, sess = extract_oasis_subject_id("OAS1_0001_MR1_mpr-1_100.jpg", 0)
    assert subj == "OAS1_0001"
    assert sess == "MR1"

    subj, sess = extract_oasis_subject_id("oas1_0123_mr2_slice.png", 5)
    assert subj == "OAS1_0123"
    assert sess == "MR2"

    subj, sess = extract_oasis_subject_id("sample_26.jpg", 42)
    assert subj == "KAG_00042"
    assert sess == "SES1"


def test_preprocess_2d_slice(tmp_path):
    # Create synthetic 2D brain slice
    h, w = 180, 180
    y, x = np.ogrid[:h, :w]
    mask = ((x - 90) ** 2 / 70**2 + (y - 90) ** 2 / 80**2) <= 1.0
    img_arr = np.zeros((h, w), dtype=np.uint8)
    img_arr[mask] = 128

    sample_img = Image.fromarray(img_arr)
    img_path = tmp_path / "valid_brain.png"
    sample_img.save(img_path)

    tensor = preprocess_2d_slice(img_path)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))
    # Background should be preserved as zero
    assert tensor[0, 0, 0] == 0.0


def test_preprocess_2d_slice_rejections(tmp_path):
    # 1. Non-existent file
    with pytest.raises(QCError, match="not found"):
        preprocess_2d_slice(tmp_path / "non_existent.jpg")

    # 2. Corrupt file
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"garbage content not an image")
    with pytest.raises(QCError, match="Cannot decode"):
        preprocess_2d_slice(corrupt_path)

    # 3. Underflow dimension
    tiny_img = Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 100)
    tiny_path = tmp_path / "tiny.png"
    tiny_img.save(tiny_path)
    with pytest.raises(QCError, match="smaller than min"):
        preprocess_2d_slice(tiny_path)

    # 4. Empty foreground
    empty_img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    empty_path = tmp_path / "empty.png"
    empty_img.save(empty_path)
    with pytest.raises(QCError, match="insufficient"):
        preprocess_2d_slice(empty_path)


def test_kaggle_pipeline_execution(tmp_path):
    # Setup mock Kaggle dataset folder
    raw_dir = tmp_path / "raw_kaggle"
    out_dir = tmp_path / "processed_kaggle"

    folders = [
        ("Non Demented", "OAS1_0001_MR1_mpr-1_100.jpg"),
        ("Non Demented", "OAS1_0002_MR1_mpr-1_100.jpg"),
        ("Very Mild Demented", "OAS1_0003_MR1_mpr-1_100.jpg"),
        ("Mild Demented", "OAS1_0004_MR1_mpr-1_100.jpg"),
        ("Moderate Demented", "OAS1_0005_MR1_mpr-1_100.jpg"),
        ("Moderate Demented", "OAS1_0006_MR1_mpr-1_100.jpg"),
    ]

    h, w = 128, 128
    y, x = np.ogrid[:h, :w]
    mask = ((x - 64) ** 2 / 50**2 + (y - 64) ** 2 / 50**2) <= 1.0
    img_arr = np.zeros((h, w), dtype=np.uint8)
    img_arr[mask] = 150
    sample_img = Image.fromarray(img_arr)

    for folder_name, file_name in folders:
        f_dir = raw_dir / folder_name
        f_dir.mkdir(parents=True, exist_ok=True)
        sample_img.save(f_dir / file_name)

    # Execute pipeline
    summary = execute_kaggle_preprocessing_pipeline(
        input_dir=raw_dir,
        output_dir=out_dir,
        cohort_name="MOCK-KAGGLE",
    )

    assert summary["Total_Images_Found"] == 6
    assert summary["Processed_Passed"] == 6
    assert summary["Rejected"] == 0
    assert summary["Class_Distribution"] == {"CN": 2, "MCI": 2, "AD": 2}

    # Verify output files
    assert (out_dir / "input_manifest.csv").is_file()
    assert (out_dir / "qc.jsonl").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "outputs.csv").is_file()

    # Verify zero subject leakage between splits
    manifest = pd.read_csv(out_dir / "input_manifest.csv")
    train_subjs = set(manifest[manifest["Split"] == "Train"]["Subject_ID"])
    val_subjs = set(manifest[manifest["Split"] == "Validation"]["Subject_ID"])
    test_subjs = set(manifest[manifest["Split"] == "Test"]["Subject_ID"])

    assert len(train_subjs.intersection(val_subjs)) == 0
    assert len(train_subjs.intersection(test_subjs)) == 0
    assert len(val_subjs.intersection(test_subjs)) == 0
