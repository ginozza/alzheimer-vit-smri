"""Repository depuration, security hygiene, traceability, and repeatability audits (S15 / E7 - EDT 8.0).

Implements automated verification for the official acceptance criterion of Deliverable E7
(Línea Base, Sección 6, pág. 4):
>> 'El informe mantiene trazabilidad con objetivos y resultados; el repositorio no
>>  contiene credenciales ni datos restringidos; la demostración es repetible.' <<
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.integration.config import IntegrationConfig
from src.integration.interface import TechnicalIntegrationInterface
from src.model.config import ViTConfig
from .config import ClosureConfig


# Patterns that indicate leaked secrets or private keys (Risk R8 / E7 criterion)
SECRET_REGEX_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("PRIVATE_KEY_HEADER", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("GITHUB_PAT_TOKEN", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "HARDCODED_KAGGLE_SECRET",
        re.compile(r"(?:KAGGLE_KEY|KAGGLE_API_TOKEN)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    ),
)

EXCLUDED_SCAN_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "alzheimer_vit_smri.egg-info",
    "processed",
    "raw",
}


@dataclass(frozen=True)
class RepositoryHygieneAuditResult:
    """Outcome of repository depuration and security hygiene scan (EDT 8.2 / R8)."""

    passed: bool
    scanned_files_count: int
    credential_violations: List[str] = field(default_factory=list)
    restricted_data_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TraceabilityAuditResult:
    """Outcome of objectives, deliverables, and documentary coverage audit (EDT 8.1 / I10)."""

    passed: bool
    total_required_artifacts: int
    verified_artifacts_count: int
    documentary_coverage_pct: float
    missing_artifacts: List[str] = field(default_factory=list)
    deliverables_status: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RepeatabilityAuditResult:
    """Outcome of repeatable demonstration check across CN, MCI, and AD (EDT 8.5)."""

    passed: bool
    seed: int
    tested_classes: Tuple[str, ...]
    max_logit_discrepancy: float
    max_probability_discrepancy: float
    untrained_notice_verified: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tested_classes"] = list(self.tested_classes)
        return data


@dataclass(frozen=True)
class ReleaseCandidateAuditReport:
    """Consolidated Week 15 Release Candidate & E7 Closure Audit Report."""

    release_candidate_version: str
    overall_passed: bool
    hygiene: RepositoryHygieneAuditResult
    traceability: TraceabilityAuditResult
    repeatability: RepeatabilityAuditResult

    def to_dict(self) -> dict:
        return {
            "release_candidate_version": self.release_candidate_version,
            "overall_passed": self.overall_passed,
            "hygiene": self.hygiene.to_dict(),
            "traceability": self.traceability.to_dict(),
            "repeatability": self.repeatability.to_dict(),
        }

    def to_json(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
        return target


def audit_repository_hygiene(
    repo_root: Union[str, Path],
    config: Optional[ClosureConfig] = None,
) -> RepositoryHygieneAuditResult:
    """Scan repository tree for credentials, secrets, or restricted clinical files (EDT 8.2)."""
    root = Path(repo_root).resolve()
    cfg = config or ClosureConfig()

    credential_violations: List[str] = []
    restricted_violations: List[str] = []
    scanned_count = 0

    for item in root.rglob("*"):
        if not item.is_file():
            continue
        rel_parts = item.relative_to(root).parts
        if any(part in EXCLUDED_SCAN_DIRS for part in rel_parts):
            continue

        scanned_count += 1
        rel_str = str(item.relative_to(root))
        suffix = item.suffix.lower()

        # Check forbidden extensions (.dcm, .pem, .key, etc.)
        if cfg.verify_no_restricted_data and suffix in cfg.forbidden_extensions:
            restricted_violations.append(f"{rel_str} (forbidden extension '{suffix}')")

        # Check raw clinical NIfTI volumes tracked outside data/processed
        if cfg.verify_no_restricted_data and (
            rel_str.endswith(".nii") or rel_str.endswith(".nii.gz")
        ):
            restricted_violations.append(f"{rel_str} (restricted NIfTI volume in repository tree)")

        # Scan text files for hardcoded credentials
        if cfg.verify_no_credentials and suffix in {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".sh", ".env"}:
            if item.name == ".env":
                credential_violations.append(f"{rel_str} (unignored .env file)")
                continue
            try:
                content = item.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for rule_name, pattern in SECRET_REGEX_PATTERNS:
                if pattern.search(content):
                    credential_violations.append(f"{rel_str} (matched {rule_name})")

    passed = (len(credential_violations) == 0) and (len(restricted_violations) == 0)
    return RepositoryHygieneAuditResult(
        passed=passed,
        scanned_files_count=scanned_count,
        credential_violations=credential_violations,
        restricted_data_violations=restricted_violations,
    )


def verify_project_traceability(
    repo_root: Union[str, Path],
    config: Optional[ClosureConfig] = None,
) -> TraceabilityAuditResult:
    """Verify presence of all mandatory configurations, docs, and verification scripts (EDT 8.1 / I10)."""
    root = Path(repo_root).resolve()
    cfg = config or ClosureConfig()

    all_required = (
        list(cfg.required_configs)
        + list(cfg.required_deliverable_docs)
        + list(cfg.required_verification_scripts)
    )
    missing: List[str] = []
    for rel_path in all_required:
        if not (root / rel_path).is_file():
            missing.append(rel_path)

    total = len(all_required)
    verified = total - len(missing)
    coverage_pct = round((float(verified) / float(total)) * 100.0, 2) if total > 0 else 0.0

    deliverables_status = {
        "E1_Plan_de_Datos": "ACEPTADO (H1 - S5)" if (root / "scripts/run_e1_verification.py").is_file() else "PENDIENTE",
        "E2_Preprocesamiento": "ACEPTADO (H2 - S7)" if (root / "configs/e2.json").is_file() else "PENDIENTE",
        "E3_Arquitectura_ViT": "ACEPTADO (H3 - S8)" if (root / "docs/architecture/E3_Especificacion_Arquitectura_ViT.md").is_file() else "PENDIENTE",
        "E4_Codigo_Fuente": "ACEPTADO (H4 - S11)" if (root / "docs/deliverables/E4_Codigo_Fuente_del_Sistema.md").is_file() else "PENDIENTE",
        "E5_Interpretabilidad": "ACEPTADO (H5 - S12)" if (root / "docs/deliverables/E5_Modulo_Interpretabilidad_Attention_Rollout.md").is_file() else "PENDIENTE",
        "E6_Prototipo_Integracion": "ACEPTADO (H6 - S14)" if (root / "docs/deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md").is_file() else "PENDIENTE",
        "E7_Cierre_del_Proyecto": "CANDIDATO FINAL (S15)" if (root / "docs/deliverables/E7_Informe_Final_y_Cierre_del_Proyecto.md").is_file() else "PENDIENTE",
    }

    return TraceabilityAuditResult(
        passed=(len(missing) == 0),
        total_required_artifacts=total,
        verified_artifacts_count=verified,
        documentary_coverage_pct=coverage_pct,
        missing_artifacts=missing,
        deliverables_status=deliverables_status,
    )


def verify_demo_repeatability(
    seed: int = 42,
    compact_model: bool = False,
) -> RepeatabilityAuditResult:
    """Verify that the technical demonstration produces bit-identical results across runs (EDT 8.5)."""
    if compact_model:
        vit_cfg = ViTConfig(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=3,
            embed_dim=48,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
        int_cfg = IntegrationConfig(image_size=224, patch_size=16, default_seed=seed)
        iface_a = TechnicalIntegrationInterface()
        iface_b = TechnicalIntegrationInterface()
        from src.integration.pipeline import IntegratedSystemPipeline
        iface_a.pipeline = IntegratedSystemPipeline(integration_config=int_cfg, vit_config=vit_cfg)
        iface_b.pipeline = IntegratedSystemPipeline(integration_config=int_cfg, vit_config=vit_cfg)
    else:
        iface_a = TechnicalIntegrationInterface()
        iface_b = TechnicalIntegrationInterface()

    classes = ("CN", "MCI", "AD")
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    notices_ok = True

    for cls_name in classes:
        res_1 = iface_a.execute_test_input(synthetic_class=cls_name, seed=seed, subject_id=f"RC_{cls_name}")
        res_2 = iface_b.execute_test_input(synthetic_class=cls_name, seed=seed, subject_id=f"RC_{cls_name}")

        if res_1.status != "SUCCESS" or res_2.status != "SUCCESS":
            return RepeatabilityAuditResult(
                passed=False,
                seed=seed,
                tested_classes=classes,
                max_logit_discrepancy=float("inf"),
                max_probability_discrepancy=float("inf"),
                untrained_notice_verified=False,
            )

        l_diff = float(np.max(np.abs(np.array(res_1.logits) - np.array(res_2.logits))))
        p_diff = float(
            np.max(
                np.abs(
                    np.array(list(res_1.probabilities.values()))
                    - np.array(list(res_2.probabilities.values()))
                )
            )
        )
        max_logit_diff = max(max_logit_diff, l_diff)
        max_prob_diff = max(max_prob_diff, p_diff)

        if (
            res_1.is_trained_model
            or res_1.generates_clinical_diagnosis
            or "NO CONTIENE UN MODELO ENTRENADO" not in res_1.untrained_model_notice.upper()
        ):
            notices_ok = False

    passed = (max_logit_diff <= 1e-6) and (max_prob_diff <= 1e-6) and notices_ok
    return RepeatabilityAuditResult(
        passed=passed,
        seed=seed,
        tested_classes=classes,
        max_logit_discrepancy=max_logit_diff,
        max_probability_discrepancy=max_prob_diff,
        untrained_notice_verified=notices_ok,
    )


def run_release_candidate_audit(
    repo_root: Union[str, Path],
    config: Optional[ClosureConfig] = None,
    compact_demo: bool = False,
) -> ReleaseCandidateAuditReport:
    """Execute the complete Week 15 Release Candidate & E7 Closure Audit."""
    cfg = config or ClosureConfig()
    hygiene = audit_repository_hygiene(repo_root, cfg)
    traceability = verify_project_traceability(repo_root, cfg)
    repeatability = verify_demo_repeatability(seed=cfg.demo_seed, compact_model=compact_demo)

    overall = hygiene.passed and traceability.passed and repeatability.passed
    return ReleaseCandidateAuditReport(
        release_candidate_version=cfg.release_candidate_version,
        overall_passed=overall,
        hygiene=hygiene,
        traceability=traceability,
        repeatability=repeatability,
    )
