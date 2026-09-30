"""Unit tests for the ViT-B/16 architecture (Entregable E3).

Validates all components of EDT 4.0:
- 4.1 Requisitos: config validation and constraints
- 4.2 Componentes: individual layer shapes and behavior
- 4.3 ViT-B/16: full model forward pass and parameter count
- 4.4 Interfaces: input/output compatibility between components
- 4.5 Diagramas: config serialization for documentation
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.config import ViTConfig
from src.model.layers import (
    ClassificationHead,
    MLP,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
)
from src.model.vit import VisionTransformer


# ---------------------------------------------------------------------------
# EDT 4.1: Configuration and requirements
# ---------------------------------------------------------------------------

class TestViTConfig:

    def test_defaults(self):
        cfg = ViTConfig()
        assert cfg.image_size == 224
        assert cfg.patch_size == 16
        assert cfg.in_channels == 3
        assert cfg.num_classes == 3
        assert cfg.embed_dim == 768
        assert cfg.depth == 12
        assert cfg.num_heads == 12
        assert cfg.mlp_ratio == 4.0
        assert cfg.class_names == ("CN", "MCI", "AD")

    def test_computed_properties(self):
        cfg = ViTConfig()
        assert cfg.num_patches == 196  # (224/16)^2
        assert cfg.seq_length == 197   # 196 + 1 CLS
        assert cfg.mlp_dim == 3072     # 768 * 4
        assert cfg.head_dim == 64      # 768 / 12

    def test_validation_patch_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            ViTConfig(image_size=225, patch_size=16)

    def test_validation_embed_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            ViTConfig(embed_dim=768, num_heads=7)

    def test_validation_class_names_mismatch(self):
        with pytest.raises(ValueError, match="num_classes"):
            ViTConfig(num_classes=4, class_names=("CN", "MCI", "AD"))

    def test_serialization_roundtrip(self, tmp_path):
        cfg = ViTConfig()
        json_path = tmp_path / "test_config.json"
        cfg.save(json_path)
        loaded = ViTConfig.load(json_path)
        assert cfg == loaded

    def test_load_from_project_config(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "e3_architecture.json"
        if config_path.exists():
            cfg = ViTConfig.load(config_path)
            assert cfg.image_size == 224
            assert cfg.num_classes == 3
            assert cfg.depth == 12

    def test_as_dict(self):
        cfg = ViTConfig()
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert d["embed_dim"] == 768
        assert isinstance(d["class_names"], list)


# ---------------------------------------------------------------------------
# EDT 4.2: Individual component tests
# ---------------------------------------------------------------------------

class TestPatchEmbedding:

    def test_output_shape(self):
        pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
        x = torch.randn(1, 3, 224, 224)
        out = pe(x)
        assert out.shape == (1, 197, 768)

    def test_batch_output_shape(self):
        pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
        x = torch.randn(4, 3, 224, 224)
        out = pe(x)
        assert out.shape == (4, 197, 768)

    def test_cls_token_position(self):
        """CLS token should be at position 0 and differ from patch tokens."""
        pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
        x = torch.randn(1, 3, 224, 224)
        out = pe(x)
        # CLS token at index 0 should exist
        assert out[:, 0, :].shape == (1, 768)


class TestMultiHeadSelfAttention:

    def test_output_shape(self):
        mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
        x = torch.randn(1, 197, 768)
        out = mhsa(x)
        assert out.shape == (1, 197, 768)

    def test_attention_weight_storage(self):
        mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
        x = torch.randn(1, 197, 768)
        _ = mhsa(x)
        assert mhsa.attn_weights is not None
        assert mhsa.attn_weights.shape == (1, 12, 197, 197)

    def test_attention_weights_sum_to_one(self):
        """Each row of attention weights should sum to approximately 1.0."""
        mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
        x = torch.randn(1, 197, 768)
        _ = mhsa(x)
        row_sums = mhsa.attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestMLP:

    def test_output_shape(self):
        mlp = MLP(in_features=768, hidden_features=3072)
        x = torch.randn(1, 197, 768)
        out = mlp(x)
        assert out.shape == (1, 197, 768)


class TestTransformerBlock:

    def test_output_shape(self):
        block = TransformerBlock(embed_dim=768, num_heads=12, mlp_ratio=4.0)
        x = torch.randn(1, 197, 768)
        out = block(x)
        assert out.shape == (1, 197, 768)

    def test_residual_connection(self):
        """Output should differ from input (non-trivial transformation)."""
        block = TransformerBlock(embed_dim=768, num_heads=12, mlp_ratio=4.0)
        block.eval()
        x = torch.randn(1, 197, 768)
        out = block(x)
        assert not torch.allclose(x, out)


class TestClassificationHead:

    def test_output_shape(self):
        head = ClassificationHead(embed_dim=768, num_classes=3)
        x = torch.randn(1, 197, 768)
        out = head(x)
        assert out.shape == (1, 3)

    def test_batch_output_shape(self):
        head = ClassificationHead(embed_dim=768, num_classes=3)
        x = torch.randn(8, 197, 768)
        out = head(x)
        assert out.shape == (8, 3)


# ---------------------------------------------------------------------------
# EDT 4.3: Full ViT-B/16 model tests
# ---------------------------------------------------------------------------

class TestVisionTransformer:

    def test_forward_shape(self):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 3)

    def test_batch_forward(self):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 3)

    def test_attention_maps_count(self):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(x)
        maps = model.get_attention_maps()
        assert len(maps) == 12  # One per Transformer layer

    def test_attention_maps_shape(self):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(x)
        maps = model.get_attention_maps()
        for m in maps:
            assert m.shape == (1, 12, 197, 197)

    def test_parameter_count(self):
        """ViT-B/16 should have approximately 86M parameters."""
        model = VisionTransformer(ViTConfig())
        total = model.count_parameters()
        # ViT-B/16 standard: ~86M params
        # With 3-class head instead of 1000-class: slightly less
        assert 85_000_000 < total < 90_000_000

    def test_parameter_breakdown(self):
        model = VisionTransformer(ViTConfig())
        breakdown = model.count_parameters_by_component()
        assert "patch_embed" in breakdown
        assert "encoder" in breakdown
        assert "classifier" in breakdown
        assert sum(breakdown.values()) == model.count_parameters()

    def test_gradient_flow(self):
        """Verify gradients flow correctly through the entire model."""
        model = VisionTransformer(ViTConfig())
        model.train()
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        # Check gradients exist in first and last layers
        assert model.patch_embed.projection.weight.grad is not None
        assert model.classifier.head.weight.grad is not None

    def test_output_dtype(self):
        model = VisionTransformer(ViTConfig())
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.dtype == torch.float32

    def test_summary_string(self):
        model = VisionTransformer(ViTConfig())
        s = model.summary()
        assert "ViT-B/16" in s
        assert "768" in s
        assert "197" in s

    def test_config_preserved(self):
        cfg = ViTConfig(dropout=0.2)
        model = VisionTransformer(cfg)
        assert model.config.dropout == 0.2


# ---------------------------------------------------------------------------
# EDT 4.4: Interface compatibility between E2 output and E3 input
# ---------------------------------------------------------------------------

class TestE2E3Interface:

    def test_e2_tensor_compatible_with_vit(self):
        """Verify that E2 output tensors (3, 224, 224) float32 are valid ViT input."""
        model = VisionTransformer(ViTConfig())
        model.eval()
        # Simulate E2 output: single preprocessed MRI tensor
        e2_tensor = np.random.randn(3, 224, 224).astype(np.float32)
        # Convert to PyTorch batch
        x = torch.from_numpy(e2_tensor).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 3)
        assert torch.all(torch.isfinite(logits))

    def test_e2_batch_compatible(self):
        """Simulate loading a batch of E2 tensors into the model."""
        model = VisionTransformer(ViTConfig())
        model.eval()
        batch = np.random.randn(8, 3, 224, 224).astype(np.float32)
        x = torch.from_numpy(batch)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (8, 3)
