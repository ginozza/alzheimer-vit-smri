"""Configuration schema for Week 15 & Deliverable E7: Project Closure & Release Candidate (EDT 8.0)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple, Union


@dataclass(frozen=True)
class ClosureConfig:
    """Configuration parameters for repository depuration, traceability, and repeatable demonstration."""

    release_candidate_version: str = "0.1.0-rc1"
    project_period: str = "2026-II"
    verify_no_credentials: bool = True
    verify_no_restricted_data: bool = True
    verify_repeatable_demo: bool = True
    demo_seed: int = 42
    forbidden_extensions: Tuple[str, ...] = (".dcm", ".ima", ".pem", ".key", ".p12", ".pfx")
    required_configs: Tuple[str, ...] = (
        "configs/e2.json",
        "configs/e3_architecture.json",
        "configs/e4_dataset.json",
        "configs/e4_inference.json",
        "configs/e4_training.json",
        "configs/e5_interpretabilidad.json",
        "configs/e6_integration.json",
        "configs/e7_closure.json",
    )
    required_deliverable_docs: Tuple[str, ...] = (
        "docs/base/Linea_Base_Proyecto_Alzheimer_ViT.pdf",
        "docs/architecture/E3_Especificacion_Arquitectura_ViT.md",
        "docs/deliverables/E4_Codigo_Fuente_del_Sistema.md",
        "docs/deliverables/E5_Modulo_Interpretabilidad_Attention_Rollout.md",
        "docs/deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md",
        "docs/deliverables/E7_Informe_Final_y_Cierre_del_Proyecto.md",
        "docs/deliverables/E7_Presentacion_y_Sustentacion.md",
    )
    required_verification_scripts: Tuple[str, ...] = (
        "scripts/run_e1_verification.py",
        "scripts/run_e2_demo.py",
        "scripts/run_e3_verification.py",
        "scripts/run_e4_verification.py",
        "scripts/run_e5_verification.py",
        "scripts/run_s13_verification.py",
        "scripts/run_e6_verification.py",
        "scripts/run_s15_verification.py",
    )

    def __post_init__(self) -> None:
        if not self.release_candidate_version or not self.release_candidate_version.strip():
            raise ValueError("release_candidate_version cannot be empty")
        if not self.required_configs:
            raise ValueError("required_configs cannot be empty")
        if not self.required_deliverable_docs:
            raise ValueError("required_deliverable_docs cannot be empty")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["forbidden_extensions"] = list(self.forbidden_extensions)
        data["required_configs"] = list(self.required_configs)
        data["required_deliverable_docs"] = list(self.required_deliverable_docs)
        data["required_verification_scripts"] = list(self.required_verification_scripts)
        return data

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClosureConfig":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Closure configuration file not found: {target}")
        with target.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        # Normalize optional capitalization from JSON
        if "release_Candidate_version" in raw and "release_candidate_version" not in raw:
            raw["release_candidate_version"] = raw.pop("release_Candidate_version")
        for key in (
            "forbidden_extensions",
            "required_configs",
            "required_deliverable_docs",
            "required_verification_scripts",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)
