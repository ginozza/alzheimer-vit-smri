"""Configuration schema for Deliverable E6: Integration Prototype & Portability (S13-S14)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple, Union


@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration for the end-to-end integration flow, technical interface, and I9 portability benchmark.

    Attributes:
        image_size: Target spatial resolution for 2.5D MRI slices (default 224).
        patch_size: ViT patch dimension (default 16).
        in_channels: Number of contiguous axial slices in the 2.5D representation (default 3).
        num_classes: Number of target diagnostic categories (default 3).
        class_names: Ordered class names ("CN", "MCI", "AD").
        default_device: Execution device ("cpu" or "cuda").
        default_alpha: Blending factor for Attention Rollout overlay in [0.0, 1.0].
        default_seed: Random seed for deterministic synthetic generation and reproducibility.
        is_trained_model: Flag explicitly stating whether loaded weights are clinically trained (must be False in 2026-II).
        enable_clinical_diagnosis: Flag explicitly prohibiting clinical diagnostic claims (must be False in 2026-II).
        portability_benchmark_runs: Number of timed iterations for the I9 portability benchmark.
        portability_warmup_runs: Number of warmup iterations prior to timing.
        max_latency_ms_cpu: Maximum expected CPU inference latency per sample in milliseconds (RNF-03: 500 ms).
        max_export_degradation_tol: Maximum acceptable numerical deviation between eager and TorchScript exported models.
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 3
    class_names: Tuple[str, ...] = ("CN", "MCI", "AD")
    default_device: str = "cpu"
    default_alpha: float = 0.5
    default_seed: int = 42
    is_trained_model: bool = False
    enable_clinical_diagnosis: bool = False
    portability_benchmark_runs: int = 5
    portability_warmup_runs: int = 1
    max_latency_ms_cpu: float = 2500.0
    max_export_degradation_tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive integers")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        if self.in_channels != 3:
            raise ValueError(f"2.5D MRI representation requires in_channels=3, got {self.in_channels}")
        if self.num_classes != 3:
            raise ValueError(f"Triclass architecture requires num_classes=3, got {self.num_classes}")
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must match num_classes ({self.num_classes})"
            )
        if not (0.0 <= self.default_alpha <= 1.0):
            raise ValueError(f"default_alpha must be in [0.0, 1.0], got {self.default_alpha}")
        if self.enable_clinical_diagnosis:
            raise ValueError(
                "enable_clinical_diagnosis cannot be True under academic scope restrictions (Línea Base 3.2 / E6)"
            )
        if self.portability_benchmark_runs < 1:
            raise ValueError("portability_benchmark_runs must be >= 1")
        if self.portability_warmup_runs < 0:
            raise ValueError("portability_warmup_runs must be >= 0")
        if self.max_latency_ms_cpu <= 0.0:
            raise ValueError("max_latency_ms_cpu must be > 0")
        if self.max_export_degradation_tol <= 0.0:
            raise ValueError("max_export_degradation_tol must be > 0")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["class_names"] = list(self.class_names)
        return data

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IntegrationConfig":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Integration configuration file not found: {target}")
        with target.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        if "class_names" in raw:
            raw["class_names"] = tuple(raw["class_names"])
        return cls(**raw)
