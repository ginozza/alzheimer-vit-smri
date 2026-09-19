"""End-to-end checks for partition safety, rejection logs and reproducibility."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_e2_demo import save_nifti, synthetic_record, synthetic_volume
from src.data.preprocessing import file_sha256
from src.data.preprocessing_batch import run_preprocessing


@pytest.fixture
def batch(tmp_path):
    affine = np.eye(4)
    affine[:3, 3] = [-8, -9, -20]
    rows = []
    for index, split in enumerate(("Train", "Validation", "Test")):
        filename = f"image_{index}.nii"
        save_nifti(tmp_path / filename, synthetic_volume(index, (16, 18, 20)), affine)
        rows.append(synthetic_record(f"SYNTH_{index}", "visit1", filename, split))
    return tmp_path, rows, affine


def run_batch(root, rows, name="out"):
    manifest = root / f"{name}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    summary = run_preprocessing(manifest, root, root / name)
    records = [json.loads(line) for line in (root / name / "qc.jsonl").read_text().splitlines()]
    return summary, records


def test_reproducibility_and_no_cross_partition_statistics(batch):
    root, rows, affine = batch
    input_hashes = {row["Image_File_Path"]: file_sha256(root / row["Image_File_Path"]) for row in rows}
    summary, first = run_batch(root, rows)
    _, repeat = run_batch(root, rows, "repeat")
    assert summary["processed"] == 3 and summary["rejected"] == 0
    assert [r["output_sha256"] for r in first] == [r["output_sha256"] for r in repeat]
    save_nifti(root / "image_2.nii", synthetic_volume(999, (16, 18, 20)) * 100, affine)
    _, changed = run_batch(root, rows, "changed_test")
    assert first[0]["output_sha256"] == changed[0]["output_sha256"]
    assert first[1]["output_sha256"] == changed[1]["output_sha256"]
    assert first[2]["output_sha256"] != changed[2]["output_sha256"]
    for row in rows[:2]:
        assert file_sha256(root / row["Image_File_Path"]) == input_hashes[row["Image_File_Path"]]
    for record in first:
        tensor = np.load(root / "out" / record["output_path"], allow_pickle=False)
        assert tensor.shape == (3, 224, 224) and np.isfinite(tensor).all()


def test_subject_leakage_rejected_before_qc_even_for_missing_image(batch):
    root, rows, _ = batch
    rows[1]["Subject_ID"] = rows[0]["Subject_ID"]
    rows[1]["Session_ID"] = "visit2"
    rows[1]["Image_File_Path"] = "missing.nii"
    with pytest.raises(ValueError, match="Subject leakage"):
        run_batch(root, rows)
    assert not list((root / "out").rglob("*.npy"))
    assert json.loads((root / "out/run.json").read_text())["status"] == "invalid_manifest"


@pytest.mark.parametrize("mutation", ["empty_id", "split", "mixed", "duplicate_session", "missing_column"])
def test_invalid_manifest_fails_globally(batch, mutation):
    root, rows, _ = batch
    if mutation == "empty_id":
        rows[0]["Subject_ID"] = ""
    elif mutation == "split":
        rows[0]["Split"] = "Development"
    elif mutation == "mixed":
        rows[0]["Sample_Kind"] = "real"
    elif mutation == "duplicate_session":
        rows.append(rows[0].copy())
    else:
        for row in rows:
            del row["Preprocessing_Provenance"]
    with pytest.raises(ValueError):
        run_batch(root, rows)
    assert not list((root / "out").rglob("*.npy"))


def test_longitudinal_rows_stay_in_one_split(batch):
    root, rows, _ = batch
    rows[1]["Subject_ID"] = rows[0]["Subject_ID"]
    rows[1]["Session_ID"] = "visit2"
    rows[1]["Split"] = "Train"
    summary, _ = run_batch(root, rows)
    assert summary["processed_by_split"] == {"Train": 2, "Test": 1}


def test_content_duplicate_across_compressions_and_splits(batch):
    root, rows, affine = batch
    save_nifti(root / "duplicate.nii.gz", synthetic_volume(0, (16, 18, 20)), affine)
    rows[2]["Image_File_Path"] = "duplicate.nii.gz"
    summary, records = run_batch(root, rows)
    assert summary["rejection_reasons"] == {"duplicate_content": 1}
    assert records[2]["status"] == "rejected"
    assert not (root / "out/test").exists()


@pytest.mark.parametrize("key,value,reason", [
    ("Diagnosis_Class", "OTHER", "diagnosis"), ("MMSE", "", "missing_metadata"),
    ("Age", "nan", "clinical_metadata"), ("Brain_Extraction_Pass", "false", "upstream_qc"),
    ("Space", "native", "space"), ("Modality", "fMRI", "modality"),
    ("Image_File_Path", "missing.nii", "missing_file"),
    ("Image_File_Path", "../outside.nii", "path"),
])
def test_bad_row_logged_without_stopping_batch(batch, key, value, reason):
    root, rows, _ = batch
    rows[1][key] = value
    summary, records = run_batch(root, rows)
    assert summary["processed"] == 2
    assert records[1]["reason"] == reason
    assert records[2]["status"] == "passed"
    outputs = pd.read_csv(root / "out/outputs.csv")
    assert list(outputs["Split"]) == ["Train", "Test"]


def test_output_reuse_never_overwrites(batch):
    root, rows, _ = batch
    run_batch(root, rows)
    before = file_sha256(root / "out/run.json")
    with pytest.raises(FileExistsError):
        run_batch(root, rows)
    assert file_sha256(root / "out/run.json") == before
