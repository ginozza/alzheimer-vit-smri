"""Inference abstractions and pipeline for ViT-B/16 sMRI prediction."""

from .predictor import CLASS_NAMES, InferencePipeline, PredictionResult

__all__ = [
    "CLASS_NAMES",
    "InferencePipeline",
    "PredictionResult",
]
