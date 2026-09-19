"""Manifest-driven E2 execution with isolated outputs and structured provenance."""

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import subprocess

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingConfig, QCError, file_sha256, generate_input, load_volume


SPLITS = {"Train", "Validation", "Test"}
IDENTITY_COLUMNS = ["Subject_ID", "Session_ID", "Cohort_Source", "Split", "Image_File_Path"]
REQUIRED_COLUMNS = IDENTITY_COLUMNS + [
    "Diagnosis_Class", "Age", "Sex", "MMSE", "CDR", "Quality_Control_Pass",
    "Modality", "Pulse_Sequence", "Space", "Template_ID", "Brain_Extraction_Pass",
    "Registration_QC_Pass", "Preprocessing_Provenance", "Sample_Kind",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_manifest(frame: pd.DataFrame):
    """Validate the entire split universe before filtering or extracting any row."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty manifest")
    for column in IDENTITY_COLUMNS:
        if frame[column].str.strip().eq("").any() or frame[column].ne(frame[column].str.strip()).any():
            raise ValueError(f"Missing or whitespace-padded identity: {column}")
    if not set(frame["Split"]).issubset(SPLITS):
        raise ValueError("Split must be Train, Validation or Test; supply one complete fold")
    if frame.groupby("Subject_ID")["Split"].nunique().gt(1).any():
        raise ValueError("Subject leakage detected before QC")
    if frame.duplicated(["Subject_ID", "Session_ID"]).any():
        raise ValueError("Duplicate subject/session identity")
    kinds = set(frame["Sample_Kind"])
    if not kinds.issubset({"synthetic", "real"}) or len(kinds) != 1:
        raise ValueError("Sample_Kind must be uniform: synthetic or real, never mixed")
    families = frame["Cohort_Source"].str.split("-").str[0]
    if frame.assign(Family=families).groupby("Subject_ID")["Family"].nunique().gt(1).any():
        raise ValueError("Subject identity reused across source families")


def _metadata_qc(row: dict):
    for key in REQUIRED_COLUMNS:
        if not str(row[key]).strip():
            raise QCError("missing_metadata", f"Missing {key}")
    if row["Diagnosis_Class"] not in {"CN", "MCI", "AD"}:
        raise QCError("diagnosis", "Diagnosis_Class must be CN, MCI or AD")
    for key, low, high in (("Age", 55, 95), ("MMSE", 0, 30), ("CDR", 0, 3)):
        try:
            value = float(row[key])
        except ValueError as error:
            raise QCError("clinical_metadata", f"Invalid {key}") from error
        if not np.isfinite(value) or not low <= value <= high:
            raise QCError("clinical_metadata", f"Invalid {key}")
    if float(row["CDR"]) not in {0, 0.5, 1, 2, 3} or row["Sex"] not in {"M", "F"}:
        raise QCError("clinical_metadata", "Invalid CDR or Sex")
    for key in ("Quality_Control_Pass", "Brain_Extraction_Pass", "Registration_QC_Pass"):
        if str(row[key]).lower() != "true":
            raise QCError("upstream_qc", f"{key} must explicitly be true")
    if row["Modality"] != "T1w_sMRI" or row["Pulse_Sequence"] not in {"MPRAGE", "IR-FSPGR", "T1-weighted 3D"}:
        raise QCError("modality", "Expected supported 3D T1w acquisition")
    if row["Space"] != "MNI152":
        raise QCError("space", "Requires a registered MNI152 derivative")
    cohorts = {"ADNI-1", "ADNI-GO", "ADNI-2", "ADNI-3", "OASIS-1", "OASIS-3"}
    if row["Sample_Kind"] == "synthetic":
        if row["Cohort_Source"] != "SYNTHETIC" or not row["Subject_ID"].startswith("SYNTH_"):
            raise QCError("provenance", "Synthetic samples must use SYNTHETIC source and SYNTH_ IDs")
    elif row["Cohort_Source"] not in cohorts:
        raise QCError("provenance", "Unsupported real cohort")


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "src").rglob("*.py"))
    sources += sorted((root / "scripts").rglob("*.py"))
    sources += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()]
    result = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "nibabel", "scikit-learn", "pyyaml")},
        "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in sources},
    }
    try:
        result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        result["git_head"], result["git_dirty"] = None, None
    return result


def _process_row(row: dict, root: Path, output: Path, config: PreprocessingConfig, seen: dict, record: dict):
    _metadata_qc(row)
    relative = Path(row["Image_File_Path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise QCError("path", "Image_File_Path must remain inside data root")
    if not path.is_file():
        raise QCError("missing_file", "Input file not found")
    try:
        before_hash = file_sha256(path)
        record["input_sha256"] = before_hash
        volume, geometry = load_volume(path, config)
        if file_sha256(path) != before_hash:
            raise QCError("input_changed", "Input changed during processing")
    except OSError as error:
        raise QCError("unreadable", str(error)) from error
    record.update(geometry)
    digest = geometry["content_sha256"]
    if digest in seen:
        raise QCError("duplicate_content", f"Canonical content duplicates manifest row {seen[digest]}")
    seen[digest] = record["row"]
    tensor, transforms = generate_input(volume, geometry, config)
    record.update(transforms)
    destination = Path(row["Split"].lower()) / f"sample_{record['row']:06d}.npy"
    (output / destination).parent.mkdir(exist_ok=True)
    with (output / destination).open("xb") as stream:
        np.save(stream, tensor, allow_pickle=False)
    record.update(output_path=destination.as_posix(), output_sha256=file_sha256(output / destination),
                  output_shape=list(tensor.shape), output_dtype=str(tensor.dtype))


def run_preprocessing(manifest: str | Path, data_root: str | Path, output_dir: str | Path,
                      config: PreprocessingConfig | None = None) -> dict:
    """Process one complete partition manifest; output_dir must not exist."""
    config = config or PreprocessingConfig()
    manifest, root, output = Path(manifest).resolve(), Path(data_root).resolve(), Path(output_dir).resolve()
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    output.mkdir(parents=True, exist_ok=False)
    resolved_config = config.as_dict()
    metadata = {
        "started_at": utc_now(), "status": "running", "config": resolved_config,
        "config_sha256": hashlib.sha256(json.dumps(resolved_config, sort_keys=True).encode()).hexdigest(),
        "manifest_sha256": file_sha256(manifest), "manifest_path": str(manifest),
        "data_root": str(root), "code": code_provenance(),
        "normalization_scope": "per_slice_within_subject; no fitted cohort parameters",
    }
    write_json(output / "run.json", metadata)
    frame.to_csv(output / "input_manifest.csv", index=False)
    try:
        validate_manifest(frame)
    except ValueError as error:
        metadata.update(status="invalid_manifest", finished_at=utc_now(), error=str(error))
        write_json(output / "run.json", metadata)
        write_json(output / "summary.json", {"status": "invalid_manifest", "total": len(frame), "processed": 0, "error": str(error)})
        raise
    records, seen = [], {}
    with (output / "qc.jsonl").open("x", encoding="utf-8") as log:
        for index, row in enumerate(frame.to_dict("records"), start=1):
            record = {key: row[key] for key in IDENTITY_COLUMNS + ["Diagnosis_Class", "Sample_Kind", "Template_ID", "Preprocessing_Provenance"]}
            record.update(row=index, timestamp=utc_now())
            try:
                _process_row(row, root, output, config, seen, record)
                record.update(status="passed", reason=None)
            except QCError as error:
                record.update(status="rejected", reason=error.reason, message=str(error))
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
    accepted = [record for record in records if record["status"] == "passed"]
    columns = IDENTITY_COLUMNS + ["Diagnosis_Class", "Sample_Kind", "output_path", "output_sha256"]
    pd.DataFrame(accepted, columns=columns).to_csv(output / "outputs.csv", index=False)
    summary = {
        "status": "completed" if len(accepted) == len(records) else "completed_with_rejections",
        "total": len(records), "processed": len(accepted), "rejected": len(records) - len(accepted),
        "rejection_reasons": dict(Counter(r["reason"] for r in records if r["status"] == "rejected")),
        "processed_by_split": dict(Counter(r["Split"] for r in accepted)),
        "processed_by_kind": dict(Counter(r["Sample_Kind"] for r in accepted)),
        "real_data_validation": "pending" if set(frame["Sample_Kind"]) == {"synthetic"} else "technical_qc_only",
    }
    write_json(output / "summary.json", summary)
    metadata.update(status=summary["status"], finished_at=utc_now())
    write_json(output / "run.json", metadata)
    return summary
