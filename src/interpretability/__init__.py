"""Interpretability package: Attention Rollout and explainability heatmaps for ViT-B/16."""

from .config import RolloutConfig
from .rollout import (
    CLINICAL_DISCLAIMER,
    AttentionRollout,
    RolloutResult,
)
from .visualization import (
    overlay_heatmap_on_slice,
    pseudo_jet_colormap,
    save_heatmap_png,
    save_overlay_png,
)

__all__ = [
    "CLINICAL_DISCLAIMER",
    "AttentionRollout",
    "RolloutConfig",
    "RolloutResult",
    "overlay_heatmap_on_slice",
    "pseudo_jet_colormap",
    "save_heatmap_png",
    "save_overlay_png",
]
