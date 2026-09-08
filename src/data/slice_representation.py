"""
2.5D slice extraction and tensor representation specification for ViT-B/16.
Implements the canonical hippocampal axial triplet in standard MNI152 space.
Seminario III - Periodo 2026-II, Universidad del Magdalena.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class Slice25DConfig:
    """
    Configuration parameters for 2.5D canonical slice extraction.
    """
    # MNI152 coordinate anchor: z = -12 mm corresponds approximately to slice index 70 in 1mm MNI space
    # Bilateral hippocampus, entorhinal cortex, and ambient cistern level
    canonical_z_index: int = 70
    slice_step: int = 1  # Contiguous slices: z - 1, z, z + 1
    target_height: int = 224
    target_width: int = 224
    channels: int = 3
    normalization: str = "z_score"  # or min_max
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)


class CanonicalTripletExtractor:
    """
    Transforms 3D structural MRI volumes into 2.5D 3-channel representations.
    Provides synthetic generators for unit testing ViT tensor interfaces without
    requiring massive raw neuroimaging file downloads.
    """

    def __init__(self, config: Optional[Slice25DConfig] = None):
        self.config = config or Slice25DConfig()

    def extract_triplet_from_volume(
        self,
        volume_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts 3 contiguous axial slices centered at the canonical z index.

        Args:
            volume_3d: 3D numpy array with shape (H, W, D) or (D, H, W) registered to MNI152.

        Returns:
            2.5D array of shape (3, target_height, target_width)
        """
        if volume_3d.ndim != 3:
            raise ValueError(f"Expected 3D volume, got array of shape {volume_3d.shape}")

        z = self.config.canonical_z_index
        step = self.config.slice_step

        # Slices at z - step, z, z + step
        z_indices = [z - step, z, z + step]

        slices = []
        for zi in z_indices:
            if zi < 0 or zi >= volume_3d.shape[2]:
                raise IndexError(
                    f"Slice index {zi} out of volume bounds (depth {volume_3d.shape[2]})"
                )
            slice_2d = volume_3d[:, :, zi].astype(np.float32)
            slice_norm = self.normalize_slice(slice_2d)
            slices.append(slice_norm)

        # Stack into 3 channels: (3, H, W)
        tensor_25d = np.stack(slices, axis=0)

        # Resize / crop to (3, 224, 224) if necessary
        if (
            tensor_25d.shape[1] != self.config.target_height
            or tensor_25d.shape[2] != self.config.target_width
        ):
            tensor_25d = self._resize_or_pad(
                tensor_25d, self.config.target_height, self.config.target_width
            )

        return tensor_25d

    def normalize_slice(self, slice_img: np.ndarray) -> np.ndarray:
        """Applies robust percentile clipping and min-max / z-score normalization."""
        # Non-zero brain mask
        brain_pixels = slice_img[slice_img > 0]
        if len(brain_pixels) == 0:
            return np.zeros_like(slice_img)

        p_low, p_high = np.percentile(brain_pixels, self.config.clip_percentiles)
        clipped = np.clip(slice_img, p_low, p_high)

        if self.config.normalization == "z_score":
            mean = np.mean(clipped[clipped > 0])
            std = np.std(clipped[clipped > 0]) + 1e-8
            normed = (clipped - mean) / std
        else:  # min_max
            normed = (clipped - p_low) / (p_high - p_low + 1e-8)

        return normed

    @staticmethod
    def _resize_or_pad(tensor: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Center-crops or zero-pads spatial dimensions to exactly (3, target_h, target_w)."""
        _, h, w = tensor.shape
        out = np.zeros((3, target_h, target_w), dtype=np.float32)

        # Compute cropping / padding offsets
        h_start = max(0, (h - target_h) // 2)
        h_end = min(h, h_start + target_h)
        w_start = max(0, (w - target_w) // 2)
        w_end = min(w, w_start + target_w)

        crop = tensor[:, h_start:h_end, w_start:w_end]
        c_h, c_w = crop.shape[1], crop.shape[2]

        out_h_start = (target_h - c_h) // 2
        out_w_start = (target_w - c_w) // 2
        out[:, out_h_start : out_h_start + c_h, out_w_start : out_w_start + c_w] = crop
        return out

    def generate_synthetic_batch(
        self, batch_size: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a synthetic batch of 2.5D inputs with ViT-B/16 compatible shapes:
        Inputs: (batch_size, 3, 224, 224)
        Labels: (batch_size,) with classes in {0: CN, 1: MCI, 2: AD}

        This enables architectural validation and unit testing without local GPUs
        or full raw medical datasets.
        """
        inputs = np.random.randn(
            batch_size, self.config.channels, self.config.target_height, self.config.target_width
        ).astype(np.float32)

        labels = np.random.choice([0, 1, 2], size=(batch_size,))
        return inputs, labels
