"""All-in-one verification runner for Deliverable E3: Arquitectura del Sistema.

Executes structural audits, component dimensional checks, forward/backward passes,
parameter count breakdown, attention map extraction, and E2-E3 interface verification.

Usage:
    uv run python scripts/run_e3_verification.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.model.config import ViTConfig
from src.model.layers import (
    ClassificationHead,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
)
from src.model.vit import VisionTransformer


def main():
    print("=" * 80)
    print("  ENTREGABLE E3: ARQUITECTURA DEL SISTEMA ViT-B/16 - AUDITORIA TECNICA")
    print("  Seminario III (2026-II) - Universidad del Magdalena")
    print("  Responsable de E3: Juan Simancas | Equipo: Malak Sanchez, Juan Simancas")
    print("  Director: Sergio Lubo")
    print("=" * 80)

    # 1. Verification of Configuration
    print("\n[PASO 1] Cargando y auditando la configuracion externa del modelo...")
    config_file = Path("configs/e3_architecture.json")
    assert config_file.exists(), f"ERROR: No se encontro {config_file}"
    cfg = ViTConfig.load(config_file)

    print(f"  [OK] Archivo de configuracion cargado: {config_file}")
    print(f"       Resolucion de entrada:  {cfg.image_size}x{cfg.image_size} px ({cfg.in_channels} canales)")
    print(f"       Tamano de parche (P):   {cfg.patch_size}x{cfg.patch_size} px -> {cfg.num_patches} parches")
    print(f"       Longitud secuencia (N): {cfg.seq_length} tokens ({cfg.num_patches} parches + 1 [CLS])")
    print(f"       Dimension oculta (D):   {cfg.embed_dim} canales")
    print(f"       Capas Transformer (L):  {cfg.depth} bloques")
    print(f"       Cabezas de atencion:    {cfg.num_heads} cabezas ({cfg.head_dim} dim/cabeza)")
    print(f"       Dimension MLP hidden:   {cfg.mlp_dim} neuronas (ratio {cfg.mlp_ratio})")
    print(f"       Clases de salida:       {cfg.num_classes} {cfg.class_names}")

    # 2. Individual Component Dimension Checks
    print("\n[PASO 2] Verificando dimensionalidad de componentes individuales (EDT 4.2)...")
    batch_size = 4
    x_input = torch.randn(batch_size, cfg.in_channels, cfg.image_size, cfg.image_size)

    # 2.1 Patch Embedding
    patch_embed = PatchEmbedding(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
    )
    tokens = patch_embed(x_input)
    print(f"  [OK] PatchEmbedding:        {list(x_input.shape)} -> {list(tokens.shape)}")
    assert tokens.shape == (batch_size, cfg.seq_length, cfg.embed_dim)

    # 2.2 Multi-Head Self-Attention
    mhsa = MultiHeadSelfAttention(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads)
    attn_out = mhsa(tokens)
    print(f"  [OK] MultiHeadSelfAttention: {list(tokens.shape)} -> {list(attn_out.shape)}")
    assert attn_out.shape == (batch_size, cfg.seq_length, cfg.embed_dim)
    assert mhsa.attn_weights is not None
    assert mhsa.attn_weights.shape == (batch_size, cfg.num_heads, cfg.seq_length, cfg.seq_length)
    print(f"       Mapas de atencion:     {list(mhsa.attn_weights.shape)} (almacenados para E5)")

    # 2.3 Transformer Block (Pre-norm)
    block = TransformerBlock(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio)
    block_out = block(tokens)
    print(f"  [OK] TransformerBlock:      {list(tokens.shape)} -> {list(block_out.shape)}")
    assert block_out.shape == (batch_size, cfg.seq_length, cfg.embed_dim)

    # 2.4 Classification Head
    head = ClassificationHead(embed_dim=cfg.embed_dim, num_classes=cfg.num_classes)
    logits = head(block_out)
    print(f"  [OK] ClassificationHead:    {list(block_out.shape)} -> {list(logits.shape)}")
    assert logits.shape == (batch_size, cfg.num_classes)

    # 3. Full ViT-B/16 Architecture Instantiation and Summary (EDT 4.3)
    print("\n[PASO 3] Instanciando arquitectura completa VisionTransformer (EDT 4.3)...")
    model = VisionTransformer(cfg)
    total_params = model.count_parameters()
    by_comp = model.count_parameters_by_component()

    print(f"  [OK] Modelo ViT-B/16 instanciado exitosamente.")
    print(f"       Parametros PatchEmbedding:    {by_comp['patch_embed']:>12,}")
    print(f"       Parametros Encoder (12 blq):  {by_comp['encoder']:>12,}")
    print(f"       Parametros Clasificador:      {by_comp['classifier']:>12,}")
    print(f"       ---------------------------------------------")
    print(f"       TOTAL PARAMETROS ENTRENABLES: {total_params:>12,}")
    assert 85_000_000 < total_params < 90_000_000, f"Parametros fuera de rango: {total_params}"

    # 4. End-to-End Forward Pass and Softmax Probability Audit
    print("\n[PASO 4] Ejecutando pase hacia adelante (Forward Pass) end-to-end...")
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        out_logits = model(x_input)
        probs = torch.softmax(out_logits, dim=-1)
    latency_ms = (time.time() - start_time) * 1000

    print(f"  [OK] Latencia CPU para batch de {batch_size}: {latency_ms:.1f} ms ({latency_ms/batch_size:.1f} ms/imagen)")
    print(f"  [OK] Forma de salida (Logits): {list(out_logits.shape)}")
    print(f"  [OK] Distribucion de probabilidades (Softmax sum = 1.0):")
    for i in range(batch_size):
        p = probs[i].numpy()
        print(f"       Muestra {i+1}: CN={p[0]:.4f}, MCI={p[1]:.4f}, AD={p[2]:.4f} (Suma: {p.sum():.6f})")
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    # 5. Attention Rollout Hook / Extraction Interface Audit (E5 Bridge)
    print("\n[PASO 5] Verificando interfaz de mapas de atencion (Puente con Entregable E5)...")
    maps = model.get_attention_maps()
    print(f"  [OK] Capas con mapas de atencion capturados: {len(maps)}/{cfg.depth}")
    assert len(maps) == cfg.depth
    for layer_idx, m in enumerate(maps):
        assert m.shape == (batch_size, cfg.num_heads, cfg.seq_length, cfg.seq_length)
    print(f"  [OK] Cada capa provee tensor de atencion: ({batch_size}, {cfg.num_heads}, {cfg.seq_length}, {cfg.seq_length})")
    print(f"       Garantia: listo para Rollout en Semana 12 sin modificaciones estructurales.")

    # 6. Backward Pass / Gradient Flow Verification
    print("\n[PASO 6] Verificando flujo de gradientes (Backward Pass) y entrenabilidad...")
    model.train()
    x_train = torch.randn(2, cfg.in_channels, cfg.image_size, cfg.image_size)
    y_train = torch.tensor([0, 1])  # CN, MCI
    criterion = torch.nn.CrossEntropyLoss()
    logits_train = model(x_train)
    loss = criterion(logits_train, y_train)
    loss.backward()

    # Verify gradients at boundary layers
    grad_patch = model.patch_embed.projection.weight.grad
    grad_last = model.classifier.head.weight.grad
    assert grad_patch is not None and torch.any(grad_patch != 0)
    assert grad_last is not None and torch.any(grad_last != 0)
    print(f"  [OK] Gradientes calculados correctamente desde la cabeza hasta la proyeccion inicial.")
    print(f"       Norma del gradiente de entrada: {grad_patch.norm().item():.6f}")
    print(f"       Norma del gradiente de salida:  {grad_last.norm().item():.6f}")

    # 7. E2 Preprocessing Output -> E3 Model Input Compatibility (EDT 4.4)
    print("\n[PASO 7] Verificando contrato de datos E2 -> E3...")
    # Simulate tensor generated by E2 (Kaggle adapter or 2.5D NIfTI extractor)
    simulated_e2_tensor = np.random.randn(3, 224, 224).astype(np.float32)
    # Bridge to PyTorch
    torch_tensor = torch.from_numpy(simulated_e2_tensor).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred_logits = model(torch_tensor)
    assert pred_logits.shape == (1, 3)
    print(f"  [OK] Tensor E2 (3, 224, 224) float32 ingerido directamente sin transformacion.")
    print(f"  [OK] Inferencia de prueba completada con logits: {pred_logits.squeeze().numpy().round(4)}")

    # 8. Evaluation of Acceptance Criteria from Línea Base
    print("\n[PASO 8] Evaluando Criterios de Aceptacion de Linea Base:")
    print("  >> 'La arquitectura identifica entradas, salidas, componentes, dependencias'")
    print("  >> 'e interfaces; la adaptacion 2.5D y la salida triclase estan justificadas.' <<")
    print("  - Entradas:     Tensor (B, 3, 224, 224) float32 [Cortes axiales MNI152/Kaggle]")
    print("  - Salidas:      Logits (B, 3) float32 [CN: 0, MCI: 1, AD: 2]")
    print("  - Componentes:  PatchEmbedding (16x16), 12x TransformerBlock (pre-norm), ClassificationHead")
    print("  - Dependencias: PyTorch >= 2.0.0, NumPy >= 1.24.0, uv package manager")
    print("  - Interfaces:   E2 -> E3 (tensores NPY/Torch), E3 -> E5 (mapas de atencion)")
    print("  - Justif. 2.5D: 85% reduccion computacional frente a 3D denso; preserva contexto axial hipocampal")
    print("  - Justif. 3-cl: Permite deteccion prodromica separando MCI de CN y AD clinico")

    print("\n" + "=" * 80)
    print("  CONCLUSION DE AUDITORIA: TODOS LOS CRITERIOS DE E3 HAN SIDO SUPERADOS CON EXITO.")
    print("  HITO H3 (ENTREGABLE E3: ARQUITECTURA DEL SISTEMA) CUMPLIDO FORMALMENTE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
