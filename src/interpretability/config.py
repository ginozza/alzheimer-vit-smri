"""Configuration dataclass for Attention Rollout interpretability analysis."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class RolloutConfig:
    """Hyperparameters and operational settings for Attention Rollout.

    Based on the formulation by Abnar & Zuidema (2020) 'Quantifying Attention
    Flow in Transformers', adapted for ViT-B/16 sMRI axial patch sequences.

    Attributes:
        head_reduction: Method to aggregate multi-head attention ('mean' or 'max').
        discard_ratio: Fraction of lowest attention values to discard (0.0 to 0.9).
        add_residual: Whether to add identity matrix representing residual connections.
        residual_weight: Weight given to the identity matrix (default 0.5 for (A + I)/2).
        image_size: Spatial resolution of input images (224).
        patch_size: Patch side length (16).
        cls_token_idx: Index of the classification token in the sequence (0).
        interpolate_mode: Interpolation algorithm for heatmap upsampling ('bilinear' or 'bicubic').
    """

    head_reduction: str = "mean"
    discard_ratio: float = 0.0
    add_residual: bool = True
    residual_weight: float = 0.5
    image_size: int = 224
    patch_size: int = 16
    cls_token_idx: int = 0
    interpolate_mode: str = "bilinear"

    def __post_init__(self) -> None:
        if self.head_reduction not in ("mean", "max"):
            raise ValueError(f"head_reduction must be 'mean' or 'max', got {self.head_reduction}")
        if not 0.0 <= self.discard_ratio < 1.0:
            raise ValueError(f"discard_ratio must be in [0.0, 1.0), got {self.discard_ratio}")
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0.0 < self.residual_weight < 1.0:
            raise ValueError(f"residual_weight must be in (0.0, 1.0), got {self.residual_weight}")
        if self.interpolate_mode not in ("bilinear", "bicubic", "nearest"):
            raise ValueError(f"Unsupported interpolate_mode: {self.interpolate_mode}")

    @property
    def grid_size(self) -> int:
        """Grid dimension: 224 // 16 = 14."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Total patch tokens: 14 * 14 = 196."""
        return self.grid_size ** 2

    @property
    def seq_length(self) -> int:
        """Total sequence tokens including [CLS]: 196 + 1 = 197."""
        return self.num_patches + 1

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.as_dict(), fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RolloutConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)
