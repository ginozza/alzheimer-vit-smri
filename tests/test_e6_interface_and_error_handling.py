"""Unit and integration tests for Week 14 (S14): Technical Interface, Error Control & E6 Acceptance."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.integration import (
    IntegrationConfig,
    IntegrationInputError,
    IntegratedSystemPipeline,
    TechnicalIntegrationInterface,
    UNTRAINED_MODEL_NOTICE,
    generate_synthetic_25d_case,
    run_cli,
)
from src.interpretability.rollout import CLINICAL_DISCLAIMER
from src.model.config import ViTConfig


class TestTechnicalIntegrationInterface:
    """Tests for Deliverable E6 technical interface, error handling, and mandatory notices."""

    @pytest.fixture()
    def fast_interface(self, tmp_path):
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
        cfg_file = tmp_path / "e6_fast.json"
        vit_file = tmp_path / "vit_fast.json"
        cfg.save(cfg_file)
        vit_cfg.save(vit_file)
        return TechnicalIntegrationInterface(config_path=cfg_file, vit_config_path=vit_file)

    def test_interface_executes_synthetic_input(self, fast_interface, tmp_path):
        res = fast_interface.execute_test_input(
            input_data=None,
            synthetic_class="AD",
            seed=42,
            subject_id="IFACE_SYNTH_01",
            output_dir=tmp_path / "synth_out",
        )
        assert res.status == "SUCCESS"
        assert res.subject_id == "IFACE_SYNTH_01"
        assert res.input_source == "synthetic"
        assert res.input_shape == (3, 224, 224)
        assert res.predicted_class_name in ("CN", "MCI", "AD")
        assert res.is_trained_model is False
        assert res.generates_clinical_diagnosis is False
        assert res.untrained_model_notice == UNTRAINED_MODEL_NOTICE
        assert res.clinical_disclaimer == CLINICAL_DISCLAIMER

    def test_interface_executes_npy_file_input(self, fast_interface, tmp_path):
        arr = generate_synthetic_25d_case(target_class="CN", seed=21, image_size=224)
        npy_path = tmp_path / "valid_sample.npy"
        np.save(npy_path, arr)

        res = fast_interface.execute_test_input(
            input_data=npy_path,
            subject_id="IFACE_FILE_01",
            output_dir=tmp_path / "file_out",
        )
        assert res.status == "SUCCESS"
        assert res.input_source == "npy_file"
        assert res.input_shape == (3, 224, 224)
        assert Path(res.saved_artifacts["result_json"]).is_file()

    def test_interface_executes_in_memory_torch_tensor(self, fast_interface):
        tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        res = fast_interface.execute_test_input(input_data=tensor, subject_id="IFACE_TORCH_01")
        assert res.status == "SUCCESS"
        assert res.input_source == "tensor"
        assert res.input_shape == (3, 224, 224)

    def test_error_control_missing_file(self, fast_interface, tmp_path):
        res = fast_interface.execute_test_input(input_data=tmp_path / "does_not_exist.npy")
        assert res.status == "ERROR"
        assert res.error_code == "FILE_NOT_FOUND"
        assert res.is_trained_model is False
        assert res.generates_clinical_diagnosis is False

    def test_error_control_unsupported_extension(self, fast_interface, tmp_path):
        bad_file = tmp_path / "sample.csv"
        bad_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        res = fast_interface.execute_test_input(input_data=bad_file)
        assert res.status == "ERROR"
        assert res.error_code == "UNSUPPORTED_FILE_EXTENSION"

    def test_error_control_corrupt_npy_file(self, fast_interface, tmp_path):
        corrupt = tmp_path / "broken.npy"
        corrupt.write_bytes(b"not_a_valid_numpy_binary_header")
        res = fast_interface.execute_test_input(input_data=corrupt)
        assert res.status == "ERROR"
        assert res.error_code == "CORRUPT_NPY_FILE"

    def test_error_control_invalid_dimensions(self, fast_interface):
        arr_2d = np.random.randn(32, 32).astype(np.float32)
        res = fast_interface.execute_test_input(input_data=arr_2d)
        assert res.status == "ERROR"
        assert res.error_code == "INVALID_DIMENSIONS"

    def test_error_control_unsupported_batch_size(self, fast_interface):
        arr_4d = np.random.randn(4, 3, 32, 32).astype(np.float32)
        res = fast_interface.execute_test_input(input_data=arr_4d)
        assert res.status == "ERROR"
        assert res.error_code == "UNSUPPORTED_BATCH_SIZE"

    def test_error_control_shape_mismatch(self, fast_interface):
        arr_wrong = np.random.randn(3, 64, 64).astype(np.float32)
        res = fast_interface.execute_test_input(input_data=arr_wrong)
        assert res.status == "ERROR"
        assert res.error_code == "SHAPE_MISMATCH"

    def test_error_control_non_finite_values(self, fast_interface):
        arr_nan = np.random.randn(3, 224, 224).astype(np.float32)
        arr_nan[1, 5, 5] = np.inf
        res = fast_interface.execute_test_input(input_data=arr_nan)
        assert res.status == "ERROR"
        assert res.error_code == "NON_FINITE_VALUES"

    def test_error_control_degenerate_constant_signal(self, fast_interface):
        arr_const = np.ones((3, 224, 224), dtype=np.float32) * 5.0
        res = fast_interface.execute_test_input(input_data=arr_const)
        assert res.status == "ERROR"
        assert res.error_code == "DEGENERATE_CONSTANT_SIGNAL"

    def test_error_control_invalid_target_class(self, fast_interface):
        res = fast_interface.execute_test_input(input_data=None, synthetic_class="INVALID_DX")
        assert res.status == "ERROR"
        assert res.error_code == "INVALID_TARGET_CLASS"

    def test_safe_mode_false_raises_integration_input_error(self):
        cfg = IntegrationConfig(image_size=224, patch_size=16)
        vit_cfg = ViTConfig(image_size=224, patch_size=16, embed_dim=48, depth=2, num_heads=4)
        pipeline = IntegratedSystemPipeline(integration_config=cfg, vit_config=vit_cfg)
        with pytest.raises(IntegrationInputError) as exc_info:
            pipeline.run_on_tensor(np.zeros((3, 224, 224), dtype=np.float32), safe_mode=False)
        assert exc_info.value.error_code == "DEGENERATE_CONSTANT_SIGNAL"

    def test_format_execution_report_includes_mandatory_notices(self, fast_interface):
        res_ok = fast_interface.execute_test_input(synthetic_class="MCI", seed=11)
        report_ok = fast_interface.format_execution_report(res_ok)
        assert "NO CONTIENE UN MODELO ENTRENADO" in report_ok.upper()
        assert "NI GENERA DIAGNOSTICOS" in report_ok.upper()
        assert "STATUS" in report_ok.upper() or "ESTADO DE EJECUCION" in report_ok.upper()

        res_err = fast_interface.execute_test_input(input_data=np.zeros((3, 224, 224), dtype=np.float32))
        report_err = fast_interface.format_execution_report(res_err)
        assert "DEGENERATE_CONSTANT_SIGNAL" in report_err
        assert "NO CONTIENE UN MODELO ENTRENADO" in report_err.upper()

    def test_cli_runner_success_and_error_exit_codes(self, tmp_path):
        cfg = IntegrationConfig(image_size=224, patch_size=16)
        cfg_file = tmp_path / "cli_cfg.json"
        cfg.save(cfg_file)

        code_ok, res_ok = run_cli([
            "--synthetic-class", "MCI",
            "--seed", "42",
            "--config", str(cfg_file),
            "--output-dir", str(tmp_path / "cli_ok"),
        ])
        assert code_ok == 0
        assert res_ok.status == "SUCCESS"

        code_err, res_err = run_cli([
            "--input-file", str(tmp_path / "missing.npy"),
            "--config", str(cfg_file),
        ])
        assert code_err == 1
        assert res_err.status == "ERROR"
        assert res_err.error_code == "FILE_NOT_FOUND"
