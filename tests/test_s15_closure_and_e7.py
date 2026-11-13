"""Unit and integration tests for Week 15 & Deliverable E7: Cierre del proyecto (`src.closure`)."""

from pathlib import Path
import pytest

from src.closure import (
    ClosureConfig,
    ReleaseCandidateAuditReport,
    RepeatabilityAuditResult,
    RepositoryHygieneAuditResult,
    TraceabilityAuditResult,
    audit_repository_hygiene,
    run_release_candidate_audit,
    verify_demo_repeatability,
    verify_project_traceability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_closure_config_loads_and_roundtrips(tmp_path: Path) -> None:
    """ClosureConfig loads from configs/e7_closure.json and roundtrips cleanly."""
    cfg_path = REPO_ROOT / "configs" / "e7_closure.json"
    cfg = ClosureConfig.load(cfg_path)
    assert cfg.release_candidate_version == "0.1.0-rc1"
    assert cfg.project_period == "2026-II"
    assert cfg.demo_seed == 42

    out_path = tmp_path / "closure_roundtrip.json"
    cfg.save(out_path)
    reloaded = ClosureConfig.load(out_path)
    assert reloaded == cfg


def test_closure_config_missing_file_raises(tmp_path: Path) -> None:
    """ClosureConfig.load raises FileNotFoundError on nonexistent path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        ClosureConfig.load(tmp_path / "nonexistent.json")


def test_closure_config_validation_rejects_invalid_values() -> None:
    """ClosureConfig raises ValueError on invalid parameters."""
    with pytest.raises(ValueError, match="release_candidate_version"):
        ClosureConfig(release_candidate_version="")
    with pytest.raises(ValueError, match="required_configs"):
        ClosureConfig(required_configs=())
    with pytest.raises(ValueError, match="required_deliverable_docs"):
        ClosureConfig(required_deliverable_docs=())


def test_repository_hygiene_passes_on_actual_repository() -> None:
    """The real repository passes the E7 hygiene audit with 0 credentials and 0 restricted clinical files."""
    cfg = ClosureConfig.load(REPO_ROOT / "configs" / "e7_closure.json")
    report = audit_repository_hygiene(REPO_ROOT, cfg)
    assert isinstance(report, RepositoryHygieneAuditResult)
    assert report.passed is True
    assert report.credential_violations == []
    assert report.restricted_data_violations == []
    assert report.scanned_files_count > 40


def test_repository_hygiene_detects_restricted_clinical_files(tmp_path: Path) -> None:
    """audit_repository_hygiene flags restricted neuroimaging extensions (.nii.gz, .dcm)."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "patient_scan.nii.gz").write_bytes(b"fake_nifti_payload")
    (tmp_path / "slice_001.dcm").write_bytes(b"fake_dicom_payload")

    report = audit_repository_hygiene(tmp_path)
    assert report.passed is False
    assert len(report.restricted_data_violations) == 2
    assert any("patient_scan.nii.gz" in item for item in report.restricted_data_violations)
    assert any("slice_001.dcm" in item for item in report.restricted_data_violations)


def test_repository_hygiene_detects_credential_files_and_inline_secrets(tmp_path: Path) -> None:
    """audit_repository_hygiene flags .env files and inline private keys / tokens."""
    (tmp_path / ".env").write_text("SECRET=123\n", encoding="utf-8")
    leak_file = tmp_path / "leaked_config.py"
    # Construct pattern dynamically so the test file itself remains clean
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    leak_file.write_text(f"{header}\nMIIEowIBAAKCAQEA...\n", encoding="utf-8")

    report = audit_repository_hygiene(tmp_path)
    assert report.passed is False
    assert len(report.credential_violations) >= 2


def test_project_traceability_passes_100_percent_on_actual_repository() -> None:
    """Indicator I10 documentary coverage reaches 100.0% (>= 90% threshold) on the actual repository."""
    cfg = ClosureConfig.load(REPO_ROOT / "configs" / "e7_closure.json")
    report = verify_project_traceability(REPO_ROOT, cfg)
    assert isinstance(report, TraceabilityAuditResult)
    assert report.passed is True
    assert report.documentary_coverage_pct == pytest.approx(100.0)
    assert report.missing_artifacts == []
    assert all("PENDIENTE" not in status for status in report.deliverables_status.values())


def test_project_traceability_detects_missing_artifacts(tmp_path: Path) -> None:
    """verify_project_traceability detects missing artifacts and computes coverage percentage accurately."""
    cfg = ClosureConfig(
        required_configs=("configs/a.json", "configs/b.json"),
        required_deliverable_docs=("docs/d1.md",),
        required_verification_scripts=("scripts/s1.py",),
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.json").write_text("{}", encoding="utf-8")

    report = verify_project_traceability(tmp_path, cfg)
    assert report.passed is False
    assert report.total_required_artifacts == 4
    assert report.verified_artifacts_count == 1
    assert report.documentary_coverage_pct == pytest.approx(25.0)
    assert len(report.missing_artifacts) == 3


def test_demo_repeatability_verifies_deterministic_execution() -> None:
    """verify_demo_repeatability confirms zero numerical drift across repeated runs."""
    report = verify_demo_repeatability(seed=42, compact_model=True)
    assert isinstance(report, RepeatabilityAuditResult)
    assert report.passed is True
    assert report.max_logit_discrepancy == 0.0
    assert report.max_probability_discrepancy == 0.0
    assert report.untrained_notice_verified is True


def test_run_release_candidate_audit_exports_complete_json(tmp_path: Path) -> None:
    """run_release_candidate_audit executes all E7 checks and exports a valid JSON report."""
    cfg = ClosureConfig.load(REPO_ROOT / "configs" / "e7_closure.json")
    out_json = tmp_path / "e7_audit.json"
    report = run_release_candidate_audit(
        repo_root=REPO_ROOT,
        config=cfg,
        compact_demo=True,
    )
    report.to_json(out_json)
    assert isinstance(report, ReleaseCandidateAuditReport)
    assert report.overall_passed is True
    assert out_json.exists()
    payload = report.to_dict()
    assert payload["release_candidate_version"] == "0.1.0-rc1"
    assert payload["overall_passed"] is True
    assert payload["traceability"]["documentary_coverage_pct"] == 100.0
