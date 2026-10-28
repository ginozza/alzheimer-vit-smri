"""Visualization and overlay generation for Attention Rollout heatmaps."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def pseudo_jet_colormap(val: np.ndarray) -> np.ndarray:
    """Vectorized pseudo-JET colormap mapping scalar [0, 1] to RGB [0, 255]."""
    val = np.clip(val, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(val * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(val * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(val * 4.0 - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap_on_slice(
    slice_2d: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a 2D attention heatmap on a 2D axial MRI slice.

    Args:
        slice_2d: 2D array of shape (H, W) or (C, H, W).
        heatmap: 2D float array of shape (H, W) in range [0, 1].
        alpha: Blending weight for the heatmap overlay (0.0 = only slice, 1.0 = only heatmap).

    Returns:
        RGB uint8 image array of shape (H, W, 3).
    """
    if slice_2d.ndim == 3:
        # Take first channel if (3, H, W)
        slice_2d = slice_2d[0]

    if slice_2d.shape != heatmap.shape:
        raise ValueError(f"Shape mismatch: slice {slice_2d.shape} vs heatmap {heatmap.shape}")

    # Normalize slice to [0, 255] uint8
    s_min = float(np.min(slice_2d))
    s_max = float(np.max(slice_2d))
    denom = (s_max - s_min) if (s_max - s_min) > 0 else 1.0
    slice_norm = ((slice_2d - s_min) / denom * 255.0).astype(np.uint8)
    slice_rgb = np.stack([slice_norm] * 3, axis=-1)

    # Convert heatmap to RGB
    heat_rgb = pseudo_jet_colormap(heatmap)

    # Blend
    blended = (1.0 - alpha) * slice_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def save_heatmap_png(heatmap: np.ndarray, path: Union[str, Path]) -> Path:
    """Save 2D float heatmap as color-mapped PNG."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rgb = pseudo_jet_colormap(heatmap)
    img = Image.fromarray(rgb)
    img.save(target)
    return target


def save_overlay_png(
    slice_2d: np.ndarray,
    heatmap: np.ndarray,
    path: Union[str, Path],
    alpha: float = 0.5,
) -> Path:
    """Save blended overlay image as PNG."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_slice(slice_2d, heatmap, alpha=alpha)
    img = Image.fromarray(overlay)
    img.save(target)
    return target
