"""Vision Transformer (ViT-B/16) for triclass Alzheimer classification.

Implements the full ViT-B/16 architecture (Dosovitskiy et al., 2020) adapted
for prodromal Alzheimer detection on structural MRI.  The model receives
preprocessed tensors of shape (B, 3, 224, 224) and produces logits for
three diagnostic classes: CN (0), MCI (1), AD (2).

This module defines the architecture only.  Training, evaluation, and
inference pipelines are implemented in subsequent deliverables (E4).
"""

from typing import List

import torch
import torch.nn as nn

from .config import ViTConfig
from .layers import (
    ClassificationHead,
    PatchEmbedding,
    TransformerBlock,
)


class VisionTransformer(nn.Module):
    """ViT-B/16 triclass classifier for Alzheimer prodromal detection.

    Data flow:
        Input image (B, 3, 224, 224)
        -> PatchEmbedding: (B, 197, 768)   [196 patches + 1 CLS token]
        -> 12x TransformerBlock: (B, 197, 768)
        -> ClassificationHead (CLS token only): (B, 3)

    The architecture stores attention weights from every Transformer layer
    so they can be retrieved later for Attention Rollout interpretability
    (Entregable E5, Semana 12).

    Args:
        config: ViTConfig instance with all architecture hyperparameters.
    """

    def __init__(self, config: ViTConfig | None = None):
        super().__init__()
        self.config = config or ViTConfig()

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.embed_dim,
            dropout=self.config.dropout,
        )

        # --- Transformer Encoder ---
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    qkv_bias=self.config.qkv_bias,
                    dropout=self.config.dropout,
                    attn_dropout=self.config.attn_dropout,
                    proj_dropout=self.config.proj_dropout,
                )
                for _ in range(self.config.depth)
            ]
        )

        # --- Classification Head ---
        self.classifier = ClassificationHead(
            embed_dim=self.config.embed_dim,
            num_classes=self.config.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from raw images to class logits.

        Args:
            x: Batch of preprocessed MRI tensors, shape (B, 3, 224, 224).

        Returns:
            Logits tensor of shape (B, num_classes) = (B, 3).
        """
        # (B, 3, 224, 224) -> (B, 197, 768)
        embeddings = self.patch_embed(x)

        # (B, 197, 768) -> (B, 197, 768) through 12 Transformer blocks
        encoded = self.encoder(embeddings)

        # (B, 197, 768) -> (B, 3) using [CLS] token
        logits = self.classifier(encoded)

        return logits

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Retrieve stored attention weight maps from all Transformer layers.

        Each attention map has shape (B, num_heads, seq_length, seq_length)
        = (B, 12, 197, 197) for ViT-B/16.

        These maps are populated after a forward pass and can be used for
        Attention Rollout interpretability analysis (E5).

        Returns:
            List of 12 attention weight tensors, one per Transformer layer.
        """
        maps = []
        for block in self.encoder:
            if hasattr(block, "attn") and block.attn.attn_weights is not None:
                maps.append(block.attn.attn_weights)
        return maps

    def count_parameters(self) -> int:
        """Count total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self) -> dict:
        """Count learnable parameters grouped by major component."""
        components = {
            "patch_embed": self.patch_embed,
            "encoder": self.encoder,
            "classifier": self.classifier,
        }
        return {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in components.items()
        }

    def summary(self) -> str:
        """Return a human-readable architecture summary string."""
        cfg = self.config
        total = self.count_parameters()
        by_comp = self.count_parameters_by_component()
        lines = [
            "=" * 60,
            "  ViT-B/16 Architecture Summary",
            "=" * 60,
            f"  Input:           ({cfg.in_channels}, {cfg.image_size}, {cfg.image_size})",
            f"  Patch size:      {cfg.patch_size} x {cfg.patch_size}",
            f"  Num patches:     {cfg.num_patches}",
            f"  Sequence length: {cfg.seq_length} (patches + [CLS])",
            f"  Embed dim (D):   {cfg.embed_dim}",
            f"  Transformer:     {cfg.depth} layers, {cfg.num_heads} heads",
            f"  MLP hidden:      {cfg.mlp_dim}",
            f"  Output classes:  {cfg.num_classes} {cfg.class_names}",
            f"  Dropout:         {cfg.dropout}",
            "-" * 60,
            f"  Parameters (total):        {total:>12,}",
            f"    Patch embedding:         {by_comp['patch_embed']:>12,}",
            f"    Transformer encoder:     {by_comp['encoder']:>12,}",
            f"    Classification head:     {by_comp['classifier']:>12,}",
            "=" * 60,
        ]
        return "\n".join(lines)
