"""Unit tests for the Attention Rollout interpretability module (Entregable E5)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from src.interpretability.config import RolloutConfig
from src.interpretability.rollout import (
    CLINICAL_DISCLAIMER,
    AttentionRollout,
    RolloutResult,
)
from src.interpretability.visualization import (
    overlay_heatmap_on_slice,
    pseudo_jet_colormap,
)
from src.model.config import ViTConfig
from src.model.vit import VisionTransformer


class MockViTWithAttention(nn.Module):
    """Mock ViT model returning deterministic attention maps and logits."""

    def __init__(self, depth: int = 12, num_heads: int = 12, seq_len: int = 197):
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.fc = nn.Linear(3 * 224 * 224, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.fc(x.flatten(1))

    def get_attention_maps(self) -> list[torch.Tensor]:
        # Return list of 12 attention tensors of shape (B, 12, 197, 197)
        # Uniform attention across all tokens
        uniform = torch.ones(1, self.num_heads, self.seq_len, self.seq_len) / self.seq_len
        return [uniform.clone() for _ in range(self.depth)]


class TestRolloutConfig:

    def test_defaults(self):
        cfg = RolloutConfig()
        assert cfg.head_reduction == "mean"
        assert cfg.discard_ratio == 0.0
        assert cfg.add_residual is True
        assert cfg.grid_size == 14
        assert cfg.num_patches == 196
        assert cfg.seq_length == 197

    def test_validation(self):
        with pytest.raises(ValueError, match="head_reduction"):
            RolloutConfig(head_reduction="sum")
        with pytest.raises(ValueError, match="discard_ratio"):
            RolloutConfig(discard_ratio=1.5)
        with pytest.raises(ValueError, match="residual_weight"):
            RolloutConfig(residual_weight=1.5)

    def test_serialization(self, tmp_path):
        cfg = RolloutConfig(discard_ratio=0.1, head_reduction="max")
        p = tmp_path / "test_rollout.json"
        cfg.save(p)
        loaded = RolloutConfig.load(p)
        assert cfg == loaded


class TestAttentionRolloutCore:

    @pytest.fixture
    def rollout_engine(self):
        return AttentionRollout(config=RolloutConfig())

    def test_aggregate_heads(self, rollout_engine):
        # (B=2, H=4, N=5, N=5)
        attn = torch.ones(2, 4, 5, 5) * 0.25
        mean_attn = rollout_engine.aggregate_heads(attn)
        assert mean_attn.shape == (2, 5, 5)
        assert torch.allclose(mean_attn, torch.ones(2, 5, 5) * 0.25)

    def test_discard_ratio(self):
        cfg = RolloutConfig(discard_ratio=0.5)
        engine = AttentionRollout(config=cfg)
        # (B=1, N=4, N=4) with values 0.1, 0.2, 0.3, 0.4 per row
        row = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        attn = row.repeat(1, 4, 1)  # (1, 4, 4)
        filtered = engine.apply_discard_ratio(attn)
        # Lowest 2 values (0.1, 0.2) should be 0.0
        assert filtered[0, 0, 0] == 0.0
        assert filtered[0, 0, 1] == 0.0
        # Row sum must be re-normalized to 1.0
        assert torch.allclose(filtered.sum(dim=-1), torch.ones(1, 4), atol=1e-5)

    def test_compute_rollout_shape_and_sum(self, rollout_engine):
        # 12 layers of uniform attention (1, 12, 197, 197)
        maps = [torch.ones(1, 12, 197, 197) / 197.0 for _ in range(12)]
        R = rollout_engine.compute_rollout(maps)
        assert R.shape == (1, 197, 197)
        # Each row should sum to ~1.0
        row_sums = R.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_extract_cls_attention(self, rollout_engine):
        R = torch.ones(1, 197, 197)
        grid = rollout_engine.extract_cls_attention(R)
        assert grid.shape == (1, 14, 14)

    def test_upsample_heatmap(self, rollout_engine):
        cls_grid = torch.rand(1, 14, 14)
        heatmap = rollout_engine.upsample_heatmap(cls_grid, target_size=(224, 224))
        assert heatmap.shape == (1, 224, 224)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


class TestAttentionRolloutIntegration:

    def test_explain_with_mock_model(self):
        model = MockViTWithAttention(depth=12)
        engine = AttentionRollout(model=model)
        x = torch.randn(1, 3, 224, 224)
        result = engine.explain_sample(x) if hasattr(engine, "explain_sample") else engine.explain_tensor(x)

        assert isinstance(result, RolloutResult)
        assert result.predicted_class_name in ("CN", "MCI", "AD")
        assert result.heatmap.shape == (1, 224, 224)
        assert result.cls_attention_grid.shape == (1, 14, 14)
        assert CLINICAL_DISCLAIMER in result.clinical_disclaimer

    def test_explain_with_real_vit_b16(self):
        # Instantiate full ViT-B/16
        vit_cfg = ViTConfig.load("configs/e3_architecture.json")
        model = VisionTransformer(vit_cfg)
        engine = AttentionRollout(model=model)

        x = torch.randn(1, 3, 224, 224)
        res = engine.explain_tensor(x)

        assert res.rollout_matrix.shape == (1, 197, 197)
        assert res.heatmap.shape == (1, 224, 224)
        assert 0.0 <= res.confidence <= 1.0


class TestVisualizationAndOverlays:

    def test_pseudo_jet_colormap(self):
        arr = np.linspace(0.0, 1.0, 10)
        rgb = pseudo_jet_colormap(arr)
        assert rgb.shape == (10, 3)
        assert rgb.dtype == np.uint8

    def test_overlay_heatmap(self):
        slice_2d = np.zeros((224, 224), dtype=np.float32)
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        overlay = overlay_heatmap_on_slice(slice_2d, heatmap, alpha=0.5)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            overlay_heatmap_on_slice(np.zeros((100, 100)), np.zeros((224, 224)))
