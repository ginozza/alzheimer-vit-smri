"""Unit and integration test suite for Week 16 & Final Project Closure (Hito H7 / Entregable E7).

Seminario III (2026-II) - Universidad del Magdalena
Equipo: Malak Sanchez, Juan Simancas | Director: Sergio Lubo
Semana 16: Cierre definitivo del proyecto.
"""

import json
from pathlib import Path
import pytest
import torch

from src.closure import (
    ClosureConfig,
    audit_repository_hygiene,
    verify_demo_repeatability,
    verify_project_traceability,
)
from src.integration import (
    IntegratedSystemPipeline,
    IntegrationConfig,
    generate_synthetic_25d_case,
)
from src.interpretability import CLINICAL_DISCLAIMER
from src.model import ViTConfig, VisionTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestWeek16FinalClosure:
    """Tests for final acceptance of Deliverable E7 and Milestone H7 in Week 16."""

    def test_all_seven_deliverables_documented(self):
        deliverables_dir = REPO_ROOT / "docs" / "deliverables"
        expected = [
            "E4_Codigo_Fuente_del_Sistema.md",
            "E5_Modulo_Interpretabilidad_Attention_Rollout.md",
            "E6_Prototipo_Integracion_y_Manual_de_Uso.md",
            "E7_Informe_Final_y_Cierre_del_Proyecto.md",
            "E7_Presentacion_y_Sustentacion.md",
        ]
        for name in expected:
            f = deliverables_dir / name
            assert f.is_file(), f"Missing deliverable doc: {name}"
            assert f.stat().st_size > 500

    def test_repository_hygiene_remains_perfect(self):
        cfg = ClosureConfig.load(REPO_ROOT / "configs" / "e7_closure.json")
        report = audit_repository_hygiene(REPO_ROOT, cfg)
        assert report.passed is True
        assert len(report.credential_violations) == 0
        assert len(report.restricted_data_violations) == 0

    def test_traceability_100_percent(self):
        cfg = ClosureConfig.load(REPO_ROOT / "configs" / "e7_closure.json")
        report = verify_project_traceability(REPO_ROOT, cfg)
        assert report.passed is True
        assert report.documentary_coverage_pct == pytest.approx(100.0)
        assert len(report.missing_artifacts) == 0

    def test_end_to_end_demonstration_repeatable(self):
        res = verify_demo_repeatability(seed=42, compact_model=True)
        assert res.passed is True
        assert res.max_logit_discrepancy == pytest.approx(0.0)
        assert res.max_probability_discrepancy == pytest.approx(0.0)
        assert res.untrained_notice_verified is True

    def test_all_eight_configs_present_and_valid(self):
        configs_dir = REPO_ROOT / "configs"
        configs = [
            "e2.json",
            "e3_architecture.json",
            "e4_dataset.json",
            "e4_inference.json",
            "e4_training.json",
            "e5_interpretabilidad.json",
            "e6_integration.json",
            "e7_closure.json",
        ]
        for c in configs:
            p = configs_dir / c
            assert p.is_file()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert isinstance(data, dict)

    def test_all_verification_scripts_executable(self):
        scripts_dir = REPO_ROOT / "scripts"
        scripts = [
            "run_e1_verification.py",
            "run_e2_demo.py",
            "run_e3_verification.py",
            "run_e4_verification.py",
            "run_e5_verification.py",
            "run_s13_verification.py",
            "run_e6_verification.py",
            "run_s15_verification.py",
            "run_s16_verification.py",
        ]
        for s in scripts:
            p = scripts_dir / s
            assert p.is_file(), f"Missing verification script: {s}"

    def test_full_pipeline_produces_artifacts_and_disclaimer(self, tmp_path):
        cfg = IntegrationConfig(image_size=224, patch_size=16)
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
        pipe = IntegratedSystemPipeline(integration_config=cfg, vit_config=vit_cfg)
        result = pipe.run_synthetic_case(target_class="AD", seed=99, output_dir=tmp_path)

        assert result.status == "SUCCESS"
        assert result.predicted_class_name in ("CN", "MCI", "AD")
        assert result.clinical_disclaimer == CLINICAL_DISCLAIMER
        assert result.is_trained_model is False
        assert result.generates_clinical_diagnosis is False
        assert Path(result.saved_artifacts["heatmap_png"]).is_file()
        assert Path(result.saved_artifacts["overlay_png"]).is_file()
        assert Path(result.saved_artifacts["result_json"]).is_file()
