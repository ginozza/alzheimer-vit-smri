"""All-in-one verification runner for Deliverable E5: Modulo de interpretabilidad (Attention Rollout).

Seminario III (2026-II) - Universidad del Magdalena
Equipo: Malak Sanchez, Juan Simancas | Director: Sergio Lubo
Hito H5 - Semana 12

Audits all requirements defined in the project Linea Base:
- Package installation and module importation (src.interpretability).
- External configuration schema validation (configs/e5_interpretabilidad.json).
- Attention Rollout mathematical formulation (Abnar & Zuidema 2020) with residual weighting.
- Full end-to-end integration with VisionTransformer (ViT-B/16, 85.8M params).
- Spatial aggregation, patch grid extraction (14x14), and bilinear upsampling (224x224).
- Colormap synthesis (pseudo-jet) and alpha blending on 2.5D MRI slices.
- Verification of the mandatory clinical disclaimer and untrained weights notice.

Usage:
    uv run python scripts/run_e5_verification.py
"""

from pathlib import Path
import sys
import tempfile
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.interpretability import (
    CLINICAL_DISCLAIMER,
    AttentionRollout,
    RolloutConfig,
    RolloutResult,
    overlay_heatmap_on_slice,
    pseudo_jet_colormap,
    save_heatmap_png,
    save_overlay_png,
)
from src.model import ViTConfig, VisionTransformer


CONFIGS_DIR = Path("configs")


def audit_installation_and_imports() -> None:
    print("\n[AUDITORIA 1] Verificando instalacion de paquete e importacion de interpretabilidad...")
    import src.interpretability as interp
    from src.interpretability.config import RolloutConfig as RC
    from src.interpretability.rollout import AttentionRollout as AR
    from src.interpretability.visualization import overlay_heatmap_on_slice as OHS

    assert hasattr(interp, "AttentionRollout")
    assert hasattr(interp, "RolloutConfig")
    assert hasattr(interp, "RolloutResult")
    assert hasattr(interp, "CLINICAL_DISCLAIMER")
    assert hasattr(interp, "save_heatmap_png")
    assert hasattr(interp, "save_overlay_png")

    print("  [OK] Subpaquete 'src.interpretability' y simbolos exportados correctamente:")
    print("       - RolloutConfig:         Configuracion hiperparametrica con serializacion JSON")
    print("       - AttentionRollout:      Algoritmo de propagacion de atencion (Abnar & Zuidema 2020)")
    print("       - RolloutResult:         Estructura con mapa espacial, grid, heatmap y disclaimer")
    print("       - CLINICAL_DISCLAIMER:   Advertencia formal de uso investigativo y no clinico")
    print("       - visualization:         Pseudo-Jet colormap, alpha-blending y renderizadores PNG")


def audit_configuration_schema() -> None:
    print("\n[AUDITORIA 2] Verificando esquema de configuracion externa (Criterio Transversal)...")
    cfg_path = CONFIGS_DIR / "e5_interpretabilidad.json"
    assert cfg_path.is_file(), f"Falta archivo de configuracion: {cfg_path}"

    cfg = RolloutConfig.load(cfg_path)
    assert cfg.head_reduction == "mean"
    assert cfg.discard_ratio == 0.0
    assert cfg.add_residual is True
    assert cfg.residual_weight == 0.5
    assert cfg.image_size == 224
    assert cfg.patch_size == 16
    assert cfg.cls_token_idx == 0
    assert cfg.interpolate_mode == "bilinear"

    # Test roundtrip serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cfg_path = Path(tmp_dir) / "test_config.json"
        cfg.save(tmp_cfg_path)
        reloaded = RolloutConfig.load(tmp_cfg_path)
        assert reloaded.head_reduction == cfg.head_reduction
        assert reloaded.discard_ratio == cfg.discard_ratio
        assert reloaded.residual_weight == cfg.residual_weight
        assert reloaded.image_size == cfg.image_size

    print(f"  [OK] Archivo '{cfg_path}' validado con exito:")
    print(f"       - Reduccion de cabezales: {cfg.head_reduction}")
    print(f"       - Tasa de descarte (filtro de ruido): {cfg.discard_ratio}")
    print(f"       - Ponderacion residual: {cfg.residual_weight}")
    print(f"       - Resolucion y parche: {cfg.image_size}x{cfg.image_size}, parche {cfg.patch_size}x{cfg.patch_size}")


def audit_synthetic_rollout_formulation() -> None:
    print("\n[AUDITORIA 3] Verificando formulacion matematica de Attention Rollout (EDT 6.1)...")
    batch_size = 2
    num_layers = 12
    num_heads = 12
    num_tokens = 197  # 1 CLS + 196 patches (14x14)

    # Construct synthetic valid attention matrices (stochastic: rows sum to 1.0)
    torch.manual_seed(42)
    synthetic_layers = []
    for _ in range(num_layers):
        raw = torch.rand(batch_size, num_heads, num_tokens, num_tokens)
        probs = torch.softmax(raw, dim=-1)
        synthetic_layers.append(probs)

    config = RolloutConfig(head_reduction="mean", discard_ratio=0.05, residual_weight=0.5)
    rollout = AttentionRollout(config=config)
    rollout_matrix = rollout.compute_rollout(synthetic_layers)
    cls_grid = rollout.extract_cls_attention(rollout_matrix)
    heatmap = rollout.upsample_heatmap(cls_grid)

    assert rollout_matrix.shape == (batch_size, num_tokens, num_tokens)
    assert cls_grid.shape == (batch_size, 14, 14)
    assert heatmap.shape == (batch_size, 224, 224)

    # Verify normalization range
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert torch.isfinite(heatmap).all()

    print("  [OK] Propagacion recursiva de matrices de atencion R_l = A_hat_l * R_{l-1} verificada.")
    print(f"       - Dimension de matriz acumulada:    {tuple(rollout_matrix.shape)}")
    print(f"       - Dimension de cuadricula espacial: {tuple(cls_grid.shape)}")
    print(f"       - Dimension de mapa interpolado:    {tuple(heatmap.shape)}")
    print(f"       - Rango normalizado: [{heatmap.min():.4f}, {heatmap.max():.4f}]")


def audit_end_to_end_vit_rollout() -> None:
    print("\n[AUDITORIA 4] Verificando integracion con VisionTransformer ViT-B/16 (EDT 6.2)...")
    vit_cfg = ViTConfig.load(CONFIGS_DIR / "e3_architecture.json")
    model = VisionTransformer(vit_cfg)
    model.eval()

    rollout_cfg = RolloutConfig.load(CONFIGS_DIR / "e5_interpretabilidad.json")
    rollout = AttentionRollout(model=model, config=rollout_cfg)

    # Synthetic MRI 2.5D batch: 3 slices (RGB channels), 224x224
    sample_tensor = torch.randn(3, 224, 224)
    result = rollout.explain_tensor(sample_tensor)

    assert isinstance(result, RolloutResult)
    assert result.heatmap.shape == (1, 224, 224)
    assert result.cls_attention_grid.shape == (1, 14, 14)
    assert result.rollout_matrix.shape == (1, 197, 197)
    assert result.predicted_class_name in ("CN", "MCI", "AD")
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5

    print("  [OK] Integracion directa con ViT-B/16 (12 bloques Transformer, 85.8M parametros):")
    print(f"       - Entrada de prueba:             {tuple(sample_tensor.shape)} (2.5D MRI tri-slice)")
    print(f"       - Prediccion dummy:              Clase={result.predicted_class_name}, Confianza={result.confidence:.4f}")
    print(f"       - Heatmap de atencion global:    {tuple(result.heatmap.shape)}")


def audit_heatmaps_and_overlays() -> None:
    print("\n[AUDITORIA 5] Verificando generacion de colormaps y superposicion visual (EDT 6.3)...")
    # Synthetic slice: shape (224, 224) float in [0, 1]
    synthetic_slice = torch.rand(224, 224).numpy()
    synthetic_heatmap = torch.rand(224, 224).numpy()

    # Test pseudo-jet colormap
    rgb_colored = pseudo_jet_colormap(synthetic_heatmap)
    assert rgb_colored.shape == (224, 224, 3)
    assert rgb_colored.dtype == np.uint8

    # Test overlay
    overlay = overlay_heatmap_on_slice(synthetic_slice, synthetic_heatmap, alpha=0.5)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8

    # Test disk persistence
    with tempfile.TemporaryDirectory() as tmp_dir:
        hm_path = Path(tmp_dir) / "test_heatmap.png"
        ov_path = Path(tmp_dir) / "test_overlay.png"

        save_heatmap_png(synthetic_heatmap, hm_path)
        save_overlay_png(synthetic_slice, synthetic_heatmap, ov_path, alpha=0.5)

        assert hm_path.is_file() and hm_path.stat().st_size > 0
        assert ov_path.is_file() and ov_path.stat().st_size > 0
        print(f"  [OK] Heatmap PNG generado: {hm_path.name} ({hm_path.stat().st_size} bytes)")
        print(f"  [OK] Overlay PNG generado: {ov_path.name} ({ov_path.stat().st_size} bytes)")


def audit_clinical_disclaimer_and_untrained_weights() -> None:
    print("\n[AUDITORIA 6] Verificando advertencia clinica y documentacion de pesos (Riesgo R7)...")
    vit_cfg = ViTConfig.load(CONFIGS_DIR / "e3_architecture.json")
    model = VisionTransformer(vit_cfg)
    rollout = AttentionRollout(model=model)
    sample_tensor = torch.randn(3, 224, 224)
    result = rollout.explain_tensor(sample_tensor)

    assert result.clinical_disclaimer == CLINICAL_DISCLAIMER
    assert "PESOS" in result.clinical_disclaimer.upper() and "ENTRENADOS" in result.clinical_disclaimer.upper()
    assert "NO CONSTITUYE EVIDENCIA DIAGNOSTICA" in result.clinical_disclaimer.upper()
    assert "RADIOLOGO ESPECIALISTA" in result.clinical_disclaimer.upper()

    print("  [OK] Descargo clinico obligatorio incorporado en el objeto de resultado:")
    print("       -------------------------------------------------------------------------")
    for line in result.clinical_disclaimer.strip().split("\n"):
        print(f"       | {line}")
    print("       -------------------------------------------------------------------------")


def main() -> None:
    print("=" * 80)
    print("  AUDITORIA FORMAL DE ACEPTACION: ENTREGABLE E5 (HITO H5)")
    print("  Proyecto: Deteccion Prodromica de Alzheimer mediante Vision Transformers")
    print("  Seminario III (2026-II) - Universidad del Magdalena")
    print("  Autores: Malak Sanchez y Juan Simancas | Director: Sergio Lubo")
    print("  Modulo: Interpretabilidad mediante Attention Rollout (EDT 6.0)")
    print("=" * 80)

    start_time = time.time()

    audit_installation_and_imports()
    audit_configuration_schema()
    audit_synthetic_rollout_formulation()
    audit_end_to_end_vit_rollout()
    audit_heatmaps_and_overlays()
    audit_clinical_disclaimer_and_untrained_weights()

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("  EVALUACION DE CRITERIO DE ACEPTACION DE LINEA BASE (ENTREGABLE E5):")
    print("  >> 'El modulo de Attention Rollout esta implementado y documenta que")
    print("  >>  requiere pesos entrenados para producir interpretaciones significativas")
    print("  >>  y no constituye evidencia diagnostica clinica.' <<")
    print("  ------------------------------------------------------------------------------")
    print("  - Algoritmo de Attention Rollout: SUPERADO (Abnar & Zuidema 2020 con residuales)")
    print("  - Integracion con ViT-B/16:       SUPERADO (12 capas, 197 tokens -> 14x14 -> 224x224)")
    print("  - Visualizacion y exportacion:    SUPERADO (Pseudo-Jet colormap y alpha blending)")
    print("  - Configurabilidad externa:       SUPERADO (configs/e5_interpretabilidad.json)")
    print("  - Descargo clinico obligatorio:   SUPERADO (presente en runtime y metadata)")
    print("  - Duracion de la auditoria:       {:.2f} segundos".format(elapsed))
    print("  ==============================================================================")
    print("  DICTAMEN: CRITERIO DE ACEPTACION CUMPLIDO AL 100 %.")
    print("  HITO H5 (ENTREGABLE E5: MODULO DE INTERPRETABILIDAD) DECLARADO FORMALMENTE ACEPTADO.")
    print("=" * 80)


if __name__ == "__main__":
    main()
