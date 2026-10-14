"""Inference pipeline and single/batch prediction abstractions for sMRI ViT-B/16."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


CLASS_NAMES: Tuple[str, ...] = ("CN", "MCI", "AD")


@dataclass(frozen=True)
class PredictionResult:
    """Diagnostic inference outcome for a single MRI axial representation."""

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]
    logits: List[float]
    attention_maps: Optional[List[List[List[List[float]]]]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2)


class InferencePipeline:
    """Encapsulates inference execution with strict validation and no_grad safety."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        class_names: Tuple[str, ...] = CLASS_NAMES,
    ):
        self.device = torch.device(device if device is not None else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names

    def _prepare_tensor(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert and validate input to shape (B, 3, 224, 224) float32."""
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            tensor = x
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        # Allow single 3D slice (3, 224, 224) -> add batch dimension
        if tensor.ndim == 3:
            if tensor.shape != (3, 224, 224):
                raise ValueError(f"Expected 3D tensor of shape (3, 224, 224), got {tuple(tensor.shape)}")
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[1:] != (3, 224, 224):
                raise ValueError(f"Expected 4D tensor of shape (B, 3, 224, 224), got {tuple(tensor.shape)}")
        else:
            raise ValueError(f"Input tensor must have 3 or 4 dimensions, got {tensor.ndim}")

        if not torch.all(torch.isfinite(tensor)):
            raise ValueError("Input tensor contains non-finite values (NaN or Inf)")

        return tensor.to(self.device)

    def predict_tensor(
        self,
        x: Union[torch.Tensor, np.ndarray],
        return_attention: bool = False,
    ) -> PredictionResult:
        """Run inference on a single 2.5D MRI tensor."""
        batch = self._prepare_tensor(x)
        if batch.shape[0] != 1:
            raise ValueError(f"predict_tensor expects a single sample, got batch size {batch.shape[0]}")

        results = self.predict_batch(batch, return_attention=return_attention)
        return results[0]

    def predict_batch(
        self,
        x: Union[torch.Tensor, np.ndarray],
        return_attention: bool = False,
    ) -> List[PredictionResult]:
        """Run inference across a batch of 2.5D MRI tensors."""
        batch = self._prepare_tensor(x)
        self.model.eval()

        with torch.no_grad():
            logits_tensor = self.model(batch)
            probs_tensor = torch.softmax(logits_tensor, dim=-1)

            attention_maps = None
            if return_attention and hasattr(self.model, "get_attention_maps"):
                raw_maps = self.model.get_attention_maps()
                # Keep attention maps on CPU as numpy/list
                attention_maps = [m.detach().cpu().tolist() for m in raw_maps]

        results: List[PredictionResult] = []
        for i in range(batch.shape[0]):
            sample_logits = logits_tensor[i].cpu().tolist()
            sample_probs = probs_tensor[i].cpu().tolist()
            pred_id = int(np.argmax(sample_probs))
            conf = float(sample_probs[pred_id])

            prob_dict = {
                name: float(sample_probs[idx])
                for idx, name in enumerate(self.class_names)
            }

            # Slice sample attention if returned
            sample_attn = None
            if attention_maps is not None:
                # attention_maps is list of 12 layers: each has shape (B, num_heads, N, N)
                sample_attn = [layer[i] for layer in attention_maps]

            results.append(
                PredictionResult(
                    predicted_class_id=pred_id,
                    predicted_class_name=self.class_names[pred_id],
                    confidence=conf,
                    probabilities=prob_dict,
                    logits=sample_logits,
                    attention_maps=sample_attn,
                )
            )

        return results

    def predict_file(
        self,
        file_path: Union[str, Path],
        return_attention: bool = False,
    ) -> PredictionResult:
        """Load a .npy tensor from disk and execute inference."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            arr = np.load(path)
        except Exception as exc:
            raise IOError(f"Failed to read numpy tensor from {path}: {exc}") from exc

        return self.predict_tensor(arr, return_attention=return_attention)
