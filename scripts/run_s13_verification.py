"""Verification runner for Week 13 (S13): Full Integrated Flow & Indicator I9 Portability Benchmark.

Seminario III (2026-II) - Universidad del Magdalena
Equipo: Malak Sanchez, Juan Simancas | Director: Sergio Lubo
Semana 13 - Avance de Entregable E6 (EDT 7.1, 7.3 e Indicador I9)

Verifies:
1. Subpackage src.integration and external config configs/e6_integration.json.
2. Deterministic synthetic 2.5D MRI triplet generation across CN, MCI, and AD.
3. End-to-end integrated execution connecting E1/E2 -> E3 -> E4 -> E5.
4. Export of structured JSON results, 224x224 Attention Rollout heatmaps, and alpha-blended overlays.
5. Indicator I9 (Portabilidad: Tamaño, latencia y degradación) via TorchScript export and CPU benchmarking.

Usage:
    uv run --locked python scripts/run_s13_verification.py
"""

from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.integration import (
    IntegrationConfig,
    IntegrationExecutionResult,
    IntegratedSystemPipeline,
    PortabilityReport,
    generate_synthetic_25d_case,
    run_portability_benchmark,
)


CONFIGS_DIR = Path("configs")


def audit_s13_configuration() -> IntegrationConfig:
    print("\n[AUDITORIA 1] Verificando configuracion externa de integracion (configs/e6_integration.json)...")
    cfg_path = CONFIGS_DIR / "e6_integration.json"
    assert cfg_path.is_file(), f"Falta archivo de configuracion: {cfg_path}"

    cfg = IntegrationConfig.load(cfg_path)
    assert cfg.image_size == 224
    assert cfg.patch_size == 16
    assert cfg.in_channels == 3
    assert cfg.num_classes == 3
    assert cfg.class_names == ("CN", "MCI", "AD")
    assert cfg.is_trained_model is False
    assert cfg.enable_clinical_diagnosis is False

    print("  [OK] Configuracion externa E6 cargada y validada:")
    print(f"       - Dimensiones 2.5D: ({cfg.in_channels}, {cfg.image_size}, {cfg.image_size})")
    print(f"       - Clases objetivo:  {cfg.class_names}")
    print(f"       - Modelo entrenado: {cfg.is_trained_model} (Restriccion Linea Base respetada)")
    return cfg


def audit_s13_synthetic_25d_generation(cfg: IntegrationConfig) -> None:
    print("\n[AUDITORIA 2] Verificando generacion determinista de casos sinteticos 2.5D (EDT 7.3)...")
    for cls_name in cfg.class_names:
        t1 = generate_synthetic_25d_case(target_class=cls_name, seed=42, image_size=cfg.image_size)
        t2 = generate_synthetic_25d_case(target_class=cls_name, seed=42, image_size=cfg.image_size)
        assert t1.shape == (3, 224, 224)
        assert t1.dtype == np.float32
        assert np.all(np.isfinite(t1))
        assert np.allclose(t1, t2), "La generacion sintetica debe ser 100 % determinista para la misma semilla"
        print(
            f"  [OK] Caso sintetico {cls_name}: shape={t1.shape}, dtype={t1.dtype}, "
            f"media={float(np.mean(t1)):.4f}, std={float(np.std(t1)):.4f}"
        )


def audit_s13_integrated_flow(cfg: IntegrationConfig) -> IntegratedSystemPipeline:
    print("\n[AUDITORIA 3] Verificando flujo completo integrado E1/E2 -> E3 -> E4 -> E5 (EDT 7.1)...")
    pipeline = IntegratedSystemPipeline(integration_config=cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cohort_results = pipeline.run_full_synthetic_cohort(
            samples_per_class=2,
            seed=42,
            output_dir=tmp_dir,
        )
        assert len(cohort_results) == 6
        for res in cohort_results:
            assert isinstance(res, IntegrationExecutionResult)
            assert res.status == "SUCCESS"
            assert res.input_shape == (3, 224, 224)
            assert res.predicted_class_name in cfg.class_names
            assert res.heatmap_shape == (224, 224)
            assert res.cls_grid_shape == (14, 14)
            assert res.is_trained_model is False
            assert res.generates_clinical_diagnosis is False
            assert Path(res.saved_artifacts["input_tensor_npy"]).is_file()
            assert Path(res.saved_artifacts["heatmap_png"]).is_file()
            assert Path(res.saved_artifacts["overlay_png"]).is_file()
            assert Path(res.saved_artifacts["result_json"]).is_file()

        summary_file = Path(tmp_dir) / "cohort_integration_summary.json"
        assert summary_file.is_file()

    print("  [OK] Cohorte sintetica de 6 sujetos (2 CN, 2 MCI, 2 AD) procesada de extremo a extremo.")
    print("  [OK] Artefactos verificados por sujeto: input_25d.npy, attention_heatmap.png, attention_overlay.png y result.json.")
    return pipeline


def audit_s13_portability_indicator_i9(
    pipeline: IntegratedSystemPipeline,
    cfg: IntegrationConfig,
) -> PortabilityReport:
    print("\n[AUDITORIA 4] Evaluando Indicador I9 de la Linea Base: Portabilidad (Tamano, Latencia y Degradacion)...")
    report = run_portability_benchmark(
        model=pipeline.model,
        integration_config=cfg,
        rollout_config=pipeline.rollout_config,
    )

    assert isinstance(report, PortabilityReport)
    assert report.parameter_count == 85_800_963
    assert report.parameter_memory_mb > 300.0
    assert report.state_dict_size_mb > 300.0
    assert report.torchscript_size_mb > 300.0
    assert report.prediction_agreement_rate == 1.0
    assert report.degradation_pct == 0.0
    assert report.degradation_within_tol is True
    assert report.latency_within_sla is True

    print(f"  [OK] Tamano de arquitectura:      {report.parameter_count:,} parametros ({report.parameter_memory_mb:.2f} MB en RAM)")
    print(f"  [OK] Tamano checkpoint (.pt):     {report.state_dict_size_mb:.2f} MB")
    print(f"  [OK] Tamano TorchScript (.pt):    {report.torchscript_size_mb:.2f} MB")
    print(f"  [OK] Latencia CPU Eager (media):  {report.eager_latency_mean_ms:.2f} ms +/- {report.eager_latency_std_ms:.2f} ms (p95={report.eager_latency_p95_ms:.2f} ms)")
    print(f"  [OK] Latencia CPU TorchScript:    {report.torchscript_latency_mean_ms:.2f} ms +/- {report.torchscript_latency_std_ms:.2f} ms")
    print(f"  [OK] Latencia Flujo Completo E2E: {report.end_to_end_latency_mean_ms:.2f} ms (Inferencia + Attention Rollout)")
    print(f"  [OK] Degradacion de exportacion:  {report.degradation_pct:.2f}% (Max |dLogit|={report.max_abs_logit_diff:.2e}, Max |dProb|={report.max_abs_prob_diff:.2e})")
    return report


def main() -> None:
    print("=" * 80)
    print("  VERIFICACION TECNICA SEMANA 13: FLUJO INTEGRADO Y PORTABILIDAD (INDICADOR I9)")
    print("  Proyecto: Deteccion Prodromica de Alzheimer mediante Vision Transformers")
    print("  Seminario III (2026-II) - Universidad del Magdalena")
    print("  Autores: Malak Sanchez y Juan Simancas | Director: Sergio Lubo")
    print("=" * 80)

    t0 = time.time()
    cfg = audit_s13_configuration()
    audit_s13_synthetic_25d_generation(cfg)
    pipeline = audit_s13_integrated_flow(cfg)
    i9_report = audit_s13_portability_indicator_i9(pipeline, cfg)
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  RESUMEN DE EVIDENCIA SEMANA 13 (CRONOGRAMA LINEA BASE):")
    print("  - Actividad S13: 'Desarrollo e integracion con el flujo completo utilizando datos sinteticos'")
    print("  - Evidencia S13: 'Flujo integrado funcional' -> CUMPLIDA AL 100 %")
    print(
        f"  - Indicador I9 (Portabilidad): Tamano={i9_report.parameter_memory_mb:.2f} MB, "
        f"Latencia={i9_report.eager_latency_mean_ms:.2f} ms, Degradacion={i9_report.degradation_pct:.2f}%"
    )
    print(f"  - Tiempo total de verificacion: {elapsed:.2f} segundos")
    print("=" * 80)


if __name__ == "__main__":
    main()
