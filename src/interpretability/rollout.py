"""Attention Rollout algorithm implementation for Vision Transformers (ViT-B/16).

Implements the recursive attention propagation framework of Abnar & Zuidema (2020)
for producing post-hoc explainability heatmaps on axial structural MRI.

NOTE ON CLINICAL APPLICABILITY:
Attention Rollout computes mathematical information flow across Transformer layers.
It requires fully trained and calibrated model weights to produce anatomically
meaningful interpretations. It does not constitute medical diagnostic evidence.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import RolloutConfig


CLINICAL_DISCLAIMER: str = (
    "ADVERTENCIA METODOLOGICA Y LIMITACION CLINICA (EDT 6.0 / R7): "
    "El mapa de Attention Rollout representa la propagacion de atencion computacional "
    "del Vision Transformer y requiere pesos formalmente entrenados y calibrados "
    "para producir patrones anatomicamente significativos. No constituye evidencia "
    "diagnostica medica, biomarcador clinico ni reemplazo de la evaluacion por un "
    "radiologo especialista."
)

CLASS_NAMES: Tuple[str, ...] = ("CN", "MCI", "AD")


@dataclass(frozen=True)
class RolloutResult:
    """Diagnostic explainability container for Attention Rollout."""

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]
    rollout_matrix: torch.Tensor
    cls_attention_grid: torch.Tensor
    heatmap: torch.Tensor
    clinical_disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "predicted_class_id": self.predicted_class_id,
            "predicted_class_name": self.predicted_class_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "heatmap_shape": list(self.heatmap.shape),
            "cls_grid_shape": list(self.cls_attention_grid.shape),
            "clinical_disclaimer": self.clinical_disclaimer,
        }

    def save_summary(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)


class AttentionRollout:
    """Computes Attention Rollout across layers of a Vision Transformer.

    Equations:
        A_bar_l = Reduction_heads(A_l)
        A_hat_l = 0.5 * A_bar_l + 0.5 * I
        R_l = A_hat_l @ R_{l-1}  with R_0 = I
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[RolloutConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config or RolloutConfig()
        self.device = torch.device(device if device is not None else "cpu")
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()

    def aggregate_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """Aggregate attention weights across heads.

        Args:
            attn: Shape (B, num_heads, N, N).

        Returns:
            Reduced tensor of shape (B, N, N).
        """
        if attn.ndim != 4:
            raise ValueError(f"Expected 4D attention tensor (B, H, N, N), got {tuple(attn.shape)}")

        if self.config.head_reduction == "mean":
            return attn.mean(dim=1)
        elif self.config.head_reduction == "max":
            return attn.max(dim=1).values
        else:
            raise ValueError(f"Unknown reduction: {self.config.head_reduction}")

    def apply_discard_ratio(self, attn: torch.Tensor) -> torch.Tensor:
        """Zero out the lowest attention weights per row and re-normalize."""
        if self.config.discard_ratio <= 0.0:
            return attn

        B, N, _ = attn.shape
        flat = attn.reshape(B * N, N)
        k = int(N * self.config.discard_ratio)
        if k <= 0:
            return attn

        # Find threshold per row
        topk_vals, _ = torch.topk(flat, k=N - k, dim=-1, largest=True)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = flat < threshold
        flat = flat.masked_fill(mask, 0.0)

        # Re-normalize row sum
        row_sums = flat.sum(dim=-1, keepdim=True) + 1e-12
        flat = flat / row_sums
        return flat.reshape(B, N, N)

    def compute_rollout(
        self,
        attention_maps: List[torch.Tensor],
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the cumulative Attention Rollout matrix.

        Args:
            attention_maps: List of L tensors of shape (B, num_heads, N, N).
            start_layer: Layer index to begin rollout from (default 0).
            end_layer: Layer index to stop rollout (inclusive, default L-1).

        Returns:
            Rollout matrix of shape (B, N, N).
        """
        if not attention_maps:
            raise ValueError("attention_maps list cannot be empty")

        total_layers = len(attention_maps)
        if end_layer is None or end_layer >= total_layers:
            end_layer = total_layers - 1

        first = attention_maps[0]
        B = first.shape[0]
        N = first.shape[-1]
        device = first.device

        # Initialize cumulative rollout with identity matrix
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
        rollout = eye.clone()

        for layer_idx in range(start_layer, end_layer + 1):
            raw_attn = attention_maps[layer_idx].to(device)
            # 1. Aggregate heads (B, N, N)
            attn = self.aggregate_heads(raw_attn)
            # 2. Optional discard of low weights
            attn = self.apply_discard_ratio(attn)
            # 3. Add identity for residual connection (A + I) / 2
            if self.config.add_residual:
                w_res = self.config.residual_weight
                attn = (1.0 - w_res) * attn + w_res * eye
                # Re-normalize row sums
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)

            # 4. Recursive matrix multiplication
            rollout = torch.bmm(attn, rollout)

        return rollout

    def extract_cls_attention(self, rollout_matrix: torch.Tensor) -> torch.Tensor:
        """Extract spatial grid attention for the [CLS] classification token.

        Args:
            rollout_matrix: Shape (B, N, N) where N = grid_size^2 + 1.

        Returns:
            Grid tensor of shape (B, grid_size, grid_size).
        """
        cls_idx = self.config.cls_token_idx
        # Extract row corresponding to CLS token: (B, N)
        cls_row = rollout_matrix[:, cls_idx, :]

        # Exclude self-attention of CLS to itself: (B, num_patches)
        if cls_idx == 0:
            patch_attn = cls_row[:, 1:]
        else:
            indices = [i for i in range(cls_row.shape[-1]) if i != cls_idx]
            patch_attn = cls_row[:, indices]

        grid = self.config.grid_size
        if patch_attn.shape[-1] != grid * grid:
            raise ValueError(
                f"Patch attention length {patch_attn.shape[-1]} does not match "
                f"grid size {grid}x{grid} ({grid * grid})"
            )

        return patch_attn.reshape(-1, grid, grid)

    def upsample_heatmap(
        self,
        cls_grid: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Interpolate (B, 14, 14) patch attention to image resolution (B, 224, 224).

        Normalizes output per sample into range [0.0, 1.0].
        """
        if target_size is None:
            target_size = (self.config.image_size, self.config.image_size)

        # Add channel dim: (B, 1, 14, 14)
        grid_4d = cls_grid.unsqueeze(1)
        align_corners = False if self.config.interpolate_mode in ("bilinear", "bicubic") else None

        upsampled = F.interpolate(
            grid_4d,
            size=target_size,
            mode=self.config.interpolate_mode,
            align_corners=align_corners,
        ).squeeze(1)  # (B, H, W)

        # Min-max normalization per sample into [0, 1]
        B = upsampled.shape[0]
        heatmaps = []
        for i in range(B):
            h = upsampled[i]
            min_val = h.min()
            max_val = h.max()
            denom = (max_val - min_val) + 1e-12
            heatmaps.append((h - min_val) / denom)

        return torch.stack(heatmaps, dim=0)

    def explain_tensor(
        self,
        x: Union[torch.Tensor, np.ndarray],
    ) -> RolloutResult:
        """Execute full end-to-end explainability pipeline for a single MRI input."""
        if self.model is None:
            raise RuntimeError("AttentionRollout was instantiated without a model")

        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
        else:
            tensor = x

        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f"Input must have 3 or 4 dimensions, got {tensor.ndim}")

        tensor = tensor.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            attention_maps = self.model.get_attention_maps()

        if not attention_maps:
            raise RuntimeError("Model returned empty attention maps. Ensure MHSA stores attention weights.")

        # Compute rollout and spatial maps
        rollout_matrix = self.compute_rollout(attention_maps)
        cls_grid = self.extract_cls_attention(rollout_matrix)
        heatmap = self.upsample_heatmap(cls_grid)

        # Build outcome
        probs_vec = probs[0].cpu().tolist()
        pred_id = int(np.argmax(probs_vec))
        prob_dict = {name: float(probs_vec[idx]) for idx, name in enumerate(CLASS_NAMES)}

        return RolloutResult(
            predicted_class_id=pred_id,
            predicted_class_name=CLASS_NAMES[pred_id],
            confidence=float(probs_vec[pred_id]),
            probabilities=prob_dict,
            rollout_matrix=rollout_matrix.cpu(),
            cls_attention_grid=cls_grid.cpu(),
            heatmap=heatmap.cpu(),
            clinical_disclaimer=CLINICAL_DISCLAIMER,
        )
