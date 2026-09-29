"""Canonical axial triplets; array inputs must already have RAS voxel axes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Slice25DConfig:
    """Array interface configuration; NIfTI callers derive the index from affine."""

    canonical_z_index: int = 70
    slice_step: int = 1
    target_height: int = 224
    target_width: int = 224
    channels: int = 3
    normalization: str = "z_score"
    clip_percentiles: tuple[float, float] = (1.0, 99.0)

    def __post_init__(self):
        for name in ("canonical_z_index", "slice_step", "target_height", "target_width", "channels"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "canonical_z_index" else 1):
                raise ValueError(f"Invalid {name}: {value}")
        if self.channels != 3 or self.slice_step != 1:
            raise ValueError("The baseline requires three contiguous slices")
        if self.normalization not in {"z_score", "min_max"}:
            raise ValueError("normalization must be z_score or min_max")
        if len(self.clip_percentiles) != 2:
            raise ValueError("Expected two clipping percentiles")
        low, high = self.clip_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("Expected 0 <= low < high <= 100")


class CanonicalTripletExtractor:
    """Extract, normalize and center crop/pad three axial slices without resampling."""

    def __init__(self, config: Slice25DConfig | None = None):
        self.config = config or Slice25DConfig()

    def extract_triplet_from_volume(self, volume_3d: np.ndarray) -> np.ndarray:
        return self.extract_with_metadata(volume_3d)[0]

    def extract_with_metadata(self, volume_3d: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Return float32 CXY data and per-slice normalization parameters."""
        if volume_3d.ndim != 3:
            raise ValueError(f"Expected 3D RAS array, got {volume_3d.shape}")
        z = self.config.canonical_z_index
        indices = [z - 1, z, z + 1]
        if indices[0] < 0 or indices[-1] >= volume_3d.shape[2]:
            raise IndexError(f"Triplet {indices} outside depth {volume_3d.shape[2]}")
        normalized = [self.normalize_with_metadata(volume_3d[:, :, index]) for index in indices]
        tensor = np.stack([item[0] for item in normalized])
        stats = [dict(item[1], z_index=index) for index, item in zip(indices, normalized)]
        return self._resize_or_pad(tensor, self.config.target_height, self.config.target_width), stats

    def normalize_slice(self, slice_img: np.ndarray) -> np.ndarray:
        return self.normalize_with_metadata(slice_img)[0]

    def normalize_with_metadata(self, slice_img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Normalize positive signal only; input must have zero background."""
        if slice_img.ndim != 2 or not np.isfinite(slice_img).all():
            raise ValueError("Expected finite 2D intensities")
        if np.any(slice_img < 0):
            raise ValueError("Expected nonnegative intensities before normalization")
        mask = slice_img > 0
        values = slice_img[mask].astype(np.float64)
        if values.size < 2:
            raise ValueError("Empty or insufficient foreground")
        low, high = np.percentile(values, self.config.clip_percentiles)
        clipped = np.clip(values, low, high)
        mean, std = float(clipped.mean()), float(clipped.std())
        if high <= low or std <= np.finfo(np.float64).eps * max(abs(mean), 1.0):
            raise ValueError("Constant foreground after clipping")
        normalized = (clipped - mean) / std if self.config.normalization == "z_score" else (clipped - low) / (high - low)
        result = np.zeros(slice_img.shape, dtype=np.float32)
        result[mask] = normalized
        return result, {
            "method": self.config.normalization,
            "foreground_voxels": int(values.size),
            "clip_low": float(low), "clip_high": float(high),
            "mean": mean, "std": std,
            "clipped_fraction": float(np.mean((values < low) | (values > high))),
        }

    @staticmethod
    def _resize_or_pad(tensor: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Center crop/pad; preserves pixel spacing and does not interpolate."""
        channels, height, width = tensor.shape
        out = np.zeros((channels, target_h, target_w), dtype=np.float32)
        size_h, size_w = min(height, target_h), min(width, target_w)
        src_h, src_w = max(0, (height - target_h) // 2), max(0, (width - target_w) // 2)
        dst_h, dst_w = max(0, (target_h - height) // 2), max(0, (target_w - width) // 2)
        out[:, dst_h:dst_h + size_h, dst_w:dst_w + size_w] = tensor[:, src_h:src_h + size_h, src_w:src_w + size_w]
        return out

    def generate_synthetic_batch(self, batch_size: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Generate deterministic interface fixtures, without clinical interpretation."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        rng = np.random.default_rng(seed)
        inputs = rng.standard_normal((batch_size, 3, self.config.target_height, self.config.target_width)).astype(np.float32)
        return inputs, rng.integers(0, 3, size=batch_size)
