"""ViT-B/16 architecture configuration with JSON serialization support."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ViTConfig:
    """Complete hyperparameter specification for ViT-B/16 triclass classifier.

    Default values correspond to the standard ViT-Base/16 architecture
    (Dosovitskiy et al., 2020) adapted for 3-class Alzheimer classification
    (CN vs MCI vs AD) on single-channel MRI slices replicated to 3 channels.

    Attributes:
        image_size: Spatial resolution of input images (square).
        patch_size: Side length of each non-overlapping patch.
        in_channels: Number of input channels (3 for replicated grayscale).
        num_classes: Number of output diagnostic classes.
        embed_dim: Dimensionality of token embeddings (D).
        depth: Number of Transformer encoder layers.
        num_heads: Number of parallel attention heads per layer.
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim.
        dropout: Dropout rate in MLP and embeddings.
        attn_dropout: Dropout rate applied to attention weights.
        proj_dropout: Dropout rate after attention output projection.
        qkv_bias: Whether to include bias in QKV linear projections.
        class_names: Ordered tuple of diagnostic class labels.
        pretrained_source: Origin of pretrained weights for future fine-tuning.
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    qkv_bias: bool = True
    class_names: Tuple[str, ...] = ("CN", "MCI", "AD")
    pretrained_source: str = "imagenet21k"

    def __post_init__(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_classes != len(self.class_names):
            raise ValueError(
                f"num_classes ({self.num_classes}) must match "
                f"len(class_names) ({len(self.class_names)})"
            )
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")

    @property
    def num_patches(self) -> int:
        """Total number of non-overlapping patches: (image_size / patch_size)^2."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_length(self) -> int:
        """Sequence length including [CLS] token: num_patches + 1."""
        return self.num_patches + 1

    @property
    def mlp_dim(self) -> int:
        """Hidden dimension of the MLP feedforward network."""
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    def as_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        d = asdict(self)
        d["class_names"] = list(d["class_names"])
        return d

    def save(self, path: str | Path) -> None:
        """Persist configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.as_dict(), fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ViTConfig":
        """Load configuration from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "class_names" in raw:
            raw["class_names"] = tuple(raw["class_names"])
        return cls(**raw)
