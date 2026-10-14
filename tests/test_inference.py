"""Unit tests for the ViT-B/16 inference pipeline and prediction structures."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from src.inference.predictor import (
    CLASS_NAMES,
    InferencePipeline,
    PredictionResult,
)


class DummyViT(nn.Module):
    """Dummy ViT model returning deterministic logits and mock attention maps."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.fill_(0.1)
            # Bias to favor class 1 (MCI)
            self.fc.bias.copy_(torch.tensor([0.2, 0.8, 0.1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)

    def get_attention_maps(self) -> list[torch.Tensor]:
        # Return 1 dummy attention layer (B, 1, 4, 4)
        return [torch.ones(1, 1, 4, 4, dtype=torch.float32) * 0.25]


class TestInferencePipeline:

    @pytest.fixture
    def pipeline(self):
        model = DummyViT()
        return InferencePipeline(model=model, device="cpu")

    def test_predict_single_tensor(self, pipeline):
        tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        res = pipeline.predict_tensor(tensor)

        assert isinstance(res, PredictionResult)
        assert res.predicted_class_id in (0, 1, 2)
        assert res.predicted_class_name in CLASS_NAMES
        assert 0.0 <= res.confidence <= 1.0
        assert len(res.logits) == 3
        assert pytest.approx(sum(res.probabilities.values())) == 1.0

    def test_predict_batch(self, pipeline):
        batch = np.random.randn(4, 3, 224, 224).astype(np.float32)
        results = pipeline.predict_batch(batch)

        assert len(results) == 4
        for r in results:
            assert isinstance(r, PredictionResult)
            assert pytest.approx(sum(r.probabilities.values())) == 1.0

    def test_predict_file(self, pipeline, tmp_path):
        tensor_path = tmp_path / "test_slice.npy"
        np.save(tensor_path, np.random.randn(3, 224, 224).astype(np.float32))

        res = pipeline.predict_file(tensor_path)
        assert isinstance(res, PredictionResult)
        assert res.predicted_class_name in CLASS_NAMES

    def test_predict_with_attention_maps(self, pipeline):
        tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        res = pipeline.predict_tensor(tensor, return_attention=True)
        assert res.attention_maps is not None
        assert len(res.attention_maps) == 1

    def test_invalid_input_rejections(self, pipeline, tmp_path):
        # 1. 2D array without channel
        with pytest.raises(ValueError, match="3 or 4 dimensions"):
            pipeline.predict_tensor(np.ones((224, 224), dtype=np.float32))

        # 2. Wrong spatial shape
        with pytest.raises(ValueError, match="Expected 3D tensor of shape"):
            pipeline.predict_tensor(np.ones((3, 100, 100), dtype=np.float32))

        # 3. Non-finite values
        bad = np.ones((3, 224, 224), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            pipeline.predict_tensor(bad)

        # 4. Missing file
        with pytest.raises(FileNotFoundError):
            pipeline.predict_file(tmp_path / "missing.npy")

    def test_serialization(self, pipeline, tmp_path):
        tensor = torch.randn(3, 224, 224, dtype=torch.float32)
        res = pipeline.predict_tensor(tensor)
        out_json = tmp_path / "pred.json"
        res.to_json(out_json)
        assert out_json.is_file()
