"""Unit and integration tests for Week 13 (S13): Full Integrated Flow & Portability (Indicator I9)."""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from src.integration import (
    IntegrationConfig,
    IntegrationExecutionResult,
    IntegratedSystemPipeline,
    PortabilityReport,
    generate_synthetic_25d_case,
    run_portability_benchmark,
)
from src.model.config import ViTConfig
from src.model.vit import VisionTransformer


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class TestIntegrationConfig:
    """Tests for IntegrationConfig validation and JSON serialization."""

    def test_default_config_values(self):
        cfg = IntegrationConfig()
        assert cfg.image_size == 224
        assert cfg.patch_size == 16
        assert cfg.in_channels == 3
        assert cfg.num_classes == 3
        assert cfg.class_names == ("CN", "MCI", "AD")
        assert cfg.is_trained_model is False
        assert cfg.enable_clinical_diagnosis is False

    def test_external_e6_config_file_loads(self):
        cfg_path = CONFIGS_DIR / "e6_integration.json"
        assert cfg_path.is_file()
        cfg = IntegrationConfig.load(cfg_path)
        assert cfg.image_size == 224
        assert cfg.class_names == ("CN", "MCI", "AD")
        assert cfg.is_trained_model is False

    def test_config_roundtrip_serialization(self, tmp_path):
        cfg = IntegrationConfig(default_alpha=0.65, default_seed=123, portability_benchmark_runs=3)
        out_file = tmp_path / "e6_test.json"
        cfg.save(out_file)
        reloaded = IntegrationConfig.load(out_file)
        assert reloaded == cfg

    def test_config_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            IntegrationConfig(image_size=225, patch_size=16)
        with pytest.raises(ValueError):
            IntegrationConfig(in_channels=1)
        with pytest.raises(ValueError):
            IntegrationConfig(num_classes=2)
        with pytest.raises(ValueError):
            IntegrationConfig(default_alpha=1.5)
        with pytest.raises(ValueError):
            IntegrationConfig(enable_clinical_diagnosis=True)


class TestSynthetic25DGeneration:
    """Tests for deterministic 2.5D MRI synthetic case generation (EDT 7.3)."""

    @pytest.mark.parametrize("cls_name", ["CN", "MCI", "AD"])
    def test_generate_synthetic_case_per_class(self, cls_name):
        tensor = generate_synthetic_25d_case(target_class=cls_name, seed=42)
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == np.float32
        assert np.all(np.isfinite(tensor))
        assert float(np.std(tensor)) > 0.1

    def test_synthetic_generation_determinism(self):
        a = generate_synthetic_25d_case(target_class="MCI", seed=77)
        b = generate_synthetic_25d_case(target_class="MCI", seed=77)
        c = generate_synthetic_25d_case(target_class="MCI", seed=78)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestIntegratedSystemPipeline:
    """Tests for end-to-end E1/E2 -> E3 -> E4 -> E5 flow (EDT 7.1)."""

    @pytest.fixture()
    def compact_pipeline(self):
        cfg = IntegrationConfig(image_size=224, patch_size=16, portability_benchmark_runs=2)
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
        return IntegratedSystemPipeline(integration_config=cfg, vit_config=vit_cfg)

    def test_run_synthetic_case_with_artifacts(self, compact_pipeline, tmp_path):
        res = compact_pipeline.run_synthetic_case(
            target_class="MCI",
            seed=42,
            subject_id="SYNTH_TEST_01",
            output_dir=tmp_path,
        )
        assert isinstance(res, IntegrationExecutionResult)
        assert res.status == "SUCCESS"
        assert res.subject_id == "SYNTH_TEST_01"
        assert res.input_shape == (3, 224, 224)
        assert res.predicted_class_name in ("CN", "MCI", "AD")
        assert 0.0 <= res.confidence <= 1.0
        assert pytest.approx(sum(res.probabilities.values()), abs=1e-4) == 1.0
        assert res.heatmap_shape == (224, 224)
        assert res.cls_grid_shape == (14, 14)
        assert res.is_trained_model is False
        assert res.generates_clinical_diagnosis is False
        assert Path(res.saved_artifacts["input_tensor_npy"]).is_file()
        assert Path(res.saved_artifacts["heatmap_png"]).is_file()
        assert Path(res.saved_artifacts["overlay_png"]).is_file()
        assert Path(res.saved_artifacts["result_json"]).is_file()

    def test_run_full_synthetic_cohort(self, compact_pipeline, tmp_path):
        results = compact_pipeline.run_full_synthetic_cohort(
            samples_per_class=2,
            seed=10,
            output_dir=tmp_path,
        )
        assert len(results) == 6
        assert all(r.status == "SUCCESS" for r in results)
        summary_file = tmp_path / "cohort_integration_summary.json"
        assert summary_file.is_file()
        summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary_data["total_cases"] == 6
        assert summary_data["successful_cases"] == 6
        assert summary_data["is_trained_model"] is False
        assert summary_data["generates_clinical_diagnosis"] is False


class TestPortabilityBenchmarkIndicatorI9:
    """Tests for Indicator I9 (Portabilidad: Tamaño, latencia y degradación)."""

    def test_portability_benchmark_compact_model(self, tmp_path):
        cfg = IntegrationConfig(
            image_size=32,
            patch_size=8,
            portability_benchmark_runs=3,
            portability_warmup_runs=1,
        )
        vit_cfg = ViTConfig(
            image_size=32,
            patch_size=8,
            in_channels=3,
            num_classes=3,
            embed_dim=48,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
        model = VisionTransformer(vit_cfg)
        report = run_portability_benchmark(
            model=model,
            integration_config=cfg,
            export_dir=tmp_path,
        )
        assert isinstance(report, PortabilityReport)
        assert report.parameter_count == model.count_parameters()
        assert report.parameter_memory_mb > 0.0
        assert report.state_dict_size_mb > 0.0
        assert report.torchscript_size_mb > 0.0
        assert report.eager_latency_mean_ms > 0.0
        assert report.torchscript_latency_mean_ms > 0.0
        assert report.end_to_end_latency_mean_ms > 0.0
        assert report.prediction_agreement_rate == 1.0
        assert report.degradation_pct == 0.0
        assert report.degradation_within_tol is True
        assert report.latency_within_sla is True

        json_out = tmp_path / "i9_report.json"
        report.to_json(json_out)
        assert json_out.is_file()
