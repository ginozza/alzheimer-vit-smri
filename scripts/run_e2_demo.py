"""Generate an explicitly synthetic E2 evidence bundle; no patient data or anatomy."""

import argparse
from pathlib import Path
import json
import struct
import zlib

import nibabel as nib
import numpy as np
import pandas as pd

from src.data.preprocessing import PreprocessingConfig, file_sha256
from src.data.preprocessing_batch import run_preprocessing
from src.data.slice_representation import CanonicalTripletExtractor


def synthetic_volume(seed: int, shape: tuple[int, int, int] = (96, 112, 96)) -> np.ndarray:
    """Deterministic ellipsoid with texture, solely for exercising technical QC."""
    rng = np.random.default_rng(seed)
    axes = np.meshgrid(*(np.linspace(-1, 1, size) for size in shape), indexing="ij")
    radius = sum(axis ** 2 for axis in axes)
    signal = 100 + 35 * axes[0] + 20 * axes[1] + 15 * axes[2] + rng.uniform(0, 12, shape)
    return np.where(radius < 0.75, signal, 0).astype(np.float32)


def save_nifti(path: Path, volume: np.ndarray, affine: np.ndarray):
    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=4)
    image.set_sform(affine, code=4)
    nib.save(image, path)


def synthetic_record(subject: str, session: str, path: str, split: str, label: str = "CN") -> dict:
    mmse, cdr = {"CN": (28, 0), "MCI": (26, 0.5), "AD": (20, 1)}[label]
    return {
        "Subject_ID": subject, "Session_ID": session, "Cohort_Source": "SYNTHETIC",
        "Diagnosis_Class": label, "Age": 70, "Sex": "F", "MMSE": mmse, "CDR": cdr,
        "Quality_Control_Pass": True, "Image_File_Path": path, "Split": split,
        "Modality": "T1w_sMRI", "Pulse_Sequence": "MPRAGE", "Space": "MNI152",
        "Template_ID": "synthetic_grid_not_an_anatomical_template",
        "Brain_Extraction_Pass": True, "Registration_QC_Pass": True,
        "Preprocessing_Provenance": "Synthetic ellipsoid; flags exercise the interface, no registration or clinical acquisition",
        "Sample_Kind": "synthetic",
    }


def write_preview(path: Path, raw: np.ndarray, processed: np.ndarray):
    """Save a six-panel grayscale PNG with fixed row-wise display ranges."""
    rows = []
    for triplet in (raw, processed):
        low, high = float(triplet.min()), float(triplet.max())
        scaled = np.rint(255 * (triplet - low) / (high - low)).astype(np.uint8)
        panels = [np.flipud(channel.T) for channel in scaled]
        rows.append(np.concatenate(panels, axis=1))
    pixels = np.concatenate(rows, axis=0)
    height, width = pixels.shape
    payload = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(payload)) + chunk(b"IEND", b""))


def create_demo(destination: Path, config: PreprocessingConfig) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    raw_dir = destination / "raw"
    raw_dir.mkdir()
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-48, -56, -48]
    rows = []
    for index, (split, label) in enumerate(zip(("Train", "Validation", "Test"), ("CN", "MCI", "AD"))):
        filename = f"synthetic_{index}.nii.gz"
        save_nifti(raw_dir / filename, synthetic_volume(42 + index), affine)
        rows.append(synthetic_record(f"SYNTH_{index}", "visit1", filename, split, label))
    save_nifti(raw_dir / "longitudinal.nii.gz", synthetic_volume(45), affine)
    rows.append(synthetic_record("SYNTH_0", "visit2", "longitudinal.nii.gz", "Train"))
    save_nifti(raw_dir / "empty.nii.gz", np.zeros((96, 112, 96), dtype=np.float32), affine)
    rows.append(synthetic_record("SYNTH_EMPTY", "visit1", "empty.nii.gz", "Train"))
    (raw_dir / "corrupt.nii.gz").write_bytes(b"deliberately invalid synthetic fixture")
    rows.append(synthetic_record("SYNTH_CORRUPT", "visit1", "corrupt.nii.gz", "Validation"))
    nonfinite = synthetic_volume(46)
    nonfinite[0, 0, 0] = np.nan
    save_nifti(raw_dir / "nan.nii.gz", nonfinite, affine)
    rows.append(synthetic_record("SYNTH_NAN", "visit1", "nan.nii.gz", "Test"))
    save_nifti(raw_dir / "duplicate.nii", synthetic_volume(42), affine)
    rows.append(synthetic_record("SYNTH_DUPLICATE", "visit1", "duplicate.nii", "Test"))
    manifest = destination / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    summary = run_preprocessing(manifest, raw_dir, destination / "processed", config)
    records = [json.loads(line) for line in (destination / "processed/qc.jsonl").read_text().splitlines()]
    first = records[0]
    if first["status"] != "passed":
        raise RuntimeError("Demonstration sample failed QC; inspect logs")
    original = nib.load(raw_dir / rows[0]["Image_File_Path"]).get_fdata(dtype=np.float32)
    raw_triplet = np.stack([original[:, :, index] for index in first["slice_indices"]])
    raw_triplet = CanonicalTripletExtractor._resize_or_pad(raw_triplet, 224, 224)
    processed = np.load(destination / "processed" / first["output_path"], allow_pickle=False)
    write_preview(destination / "preview.png", raw_triplet, processed)
    (destination / "preview.md").write_text(
        "# Synthetic technical example\n\nNo MRI acquisition, no clinical anatomy or diagnostic result.\n"
        "Top row: input slices; bottom row: normalized/padded output.\n"
        "Columns: inferior, center, superior. View uses transpose and vertical flip only for display.\n"
        "Grayscale range is shared within each row; negative z-scores make zero background gray.\n"
        "The stored tensor background remains exactly zero.\n"
        f"\nOutput SHA256: {file_sha256(destination / 'processed' / first['output_path'])}\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/e2.json")
    args = parser.parse_args()
    summary = create_demo(Path(args.output_dir), PreprocessingConfig.load(args.config))
    print(json.dumps(summary, indent=2))
    expected = {"foreground": 1, "unreadable": 1, "nonfinite": 1, "duplicate_content": 1}
    return 0 if summary["processed"] == 4 and summary["rejection_reasons"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
