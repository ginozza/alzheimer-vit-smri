"""Portability benchmark and export verification module (Indicator I9 - Semana 13).

Implements formal measurement of the three dimensions defined in the Línea Base
for Indicator I9 (Portabilidad: Tamaño, latencia y degradación):
1. Tamaño (Size): Parameter count, in-memory float32 footprint (MB), state_dict
   checkpoint size (MB), and exported TorchScript artifact size (MB).
2. Latencia (Latency): CPU execution latency per 2.5D MRI sample (mean, std, p95)
   for both PyTorch Eager and TorchScript exported models, plus end-to-end flow.
3. Degradación (Degradation): Numerical discrepancy (max absolute difference in
   logits and Softmax probabilities) and class prediction agreement rate between
   the reference PyTorch Eager model and its exported TorchScript representation.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Optional, Union
import warnings

import numpy as np
import torch

from src.interpretability.config import RolloutConfig
from src.interpretability.rollout import AttentionRollout
from src.model.config import ViTConfig
from src.model.vit import VisionTransformer
from .config import IntegrationConfig


@dataclass(frozen=True)
class PortabilityReport:
    """Structured benchmark results for Indicator I9 (Tamaño, latencia y degradación)."""

    parameter_count: int
    parameter_memory_mb: float
    state_dict_size_mb: float
    torchscript_size_mb: float
    eager_latency_mean_ms: float
    eager_latency_std_ms: float
    eager_latency_p95_ms: float
    torchscript_latency_mean_ms: float
    torchscript_latency_std_ms: float
    end_to_end_latency_mean_ms: float
    max_abs_logit_diff: float
    max_abs_prob_diff: float
    prediction_agreement_rate: float
    degradation_pct: float
    latency_within_sla: bool
    degradation_within_tol: bool
    device: str
    num_runs: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
        return target


class _LogitsOnlyWrapper(torch.nn.Module):
    """Wrapper around VisionTransformer for clean TorchScript tracing of forward logits."""

    def __init__(self, model: VisionTransformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def run_portability_benchmark(
    model: Optional[VisionTransformer] = None,
    integration_config: Optional[IntegrationConfig] = None,
    rollout_config: Optional[RolloutConfig] = None,
    export_dir: Optional[Union[str, Path]] = None,
) -> PortabilityReport:
    """Execute the Indicator I9 portability benchmark (size, latency, degradation).

    Args:
        model: Instantiated VisionTransformer (defaults to standard ViT-B/16).
        integration_config: IntegrationConfig controlling benchmark runs and thresholds.
        rollout_config: RolloutConfig for the end-to-end interpretability latency measure.
        export_dir: Optional directory to persist the exported artifacts. If None, uses a temporary dir.

    Returns:
        PortabilityReport containing verified size, latency, and degradation metrics.
    """
    cfg = integration_config or IntegrationConfig()
    r_cfg = rollout_config or RolloutConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )
    device = torch.device(cfg.default_device)

    if model is None:
        vit_cfg = ViTConfig(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            class_names=cfg.class_names,
        )
        model = VisionTransformer(vit_cfg)

    model = model.to(device)
    model.eval()

    # 1. Size metrics (Tamaño)
    param_count = model.count_parameters()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_memory_mb = float(param_bytes) / (1024.0 * 1024.0)

    torch.manual_seed(cfg.default_seed)
    sample_input = torch.randn(
        1, cfg.in_channels, cfg.image_size, cfg.image_size,
        dtype=torch.float32, device=device,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(export_dir) if export_dir is not None else Path(tmp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        state_path = work_dir / "vit_b16_state_dict.pt"
        ts_path = work_dir / "vit_b16_torchscript.pt"

        torch.save(model.state_dict(), state_path)
        state_dict_size_mb = float(state_path.stat().st_size) / (1024.0 * 1024.0)

        wrapper = _LogitsOnlyWrapper(model)
        wrapper.eval()
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            traced_model = torch.jit.trace(wrapper, sample_input, check_trace=False)
            torch.jit.save(traced_model, str(ts_path))
            loaded_ts_model = torch.jit.load(str(ts_path), map_location=device)
            loaded_ts_model.eval()

        torchscript_size_mb = float(ts_path.stat().st_size) / (1024.0 * 1024.0)

        # 2. Warmup runs
        rollout_engine = AttentionRollout(model=model, config=r_cfg, device=device)
        with torch.no_grad():
            for _ in range(cfg.portability_warmup_runs):
                _ = model(sample_input)
                _ = loaded_ts_model(sample_input)
                _ = rollout_engine.explain_tensor(sample_input)

        # 3. Latency & Degradation benchmark runs
        eager_times_ms = []
        ts_times_ms = []
        e2e_times_ms = []
        logit_diffs = []
        prob_diffs = []
        agreements = 0

        for run_idx in range(cfg.portability_benchmark_runs):
            torch.manual_seed(cfg.default_seed + run_idx + 1)
            test_x = torch.randn(
                1, cfg.in_channels, cfg.image_size, cfg.image_size,
                dtype=torch.float32, device=device,
            )

            with torch.no_grad():
                t0 = time.perf_counter()
                eager_logits = model(test_x)
                t1 = time.perf_counter()
                eager_times_ms.append((t1 - t0) * 1000.0)

                t2 = time.perf_counter()
                ts_logits = loaded_ts_model(test_x)
                t3 = time.perf_counter()
                ts_times_ms.append((t3 - t2) * 1000.0)

                t4 = time.perf_counter()
                _ = rollout_engine.explain_tensor(test_x)
                t5 = time.perf_counter()
                e2e_times_ms.append((t5 - t4) * 1000.0)

                eager_probs = torch.softmax(eager_logits, dim=-1)
                ts_probs = torch.softmax(ts_logits, dim=-1)

                logit_diff = float(torch.max(torch.abs(eager_logits - ts_logits)).item())
                prob_diff = float(torch.max(torch.abs(eager_probs - ts_probs)).item())
                logit_diffs.append(logit_diff)
                prob_diffs.append(prob_diff)

                eager_pred = int(torch.argmax(eager_probs, dim=-1).item())
                ts_pred = int(torch.argmax(ts_probs, dim=-1).item())
                if eager_pred == ts_pred:
                    agreements += 1

    eager_mean = float(np.mean(eager_times_ms))
    eager_std = float(np.std(eager_times_ms))
    eager_p95 = float(np.percentile(eager_times_ms, 95))
    ts_mean = float(np.mean(ts_times_ms))
    ts_std = float(np.std(ts_times_ms))
    e2e_mean = float(np.mean(e2e_times_ms))

    max_logit_diff = float(max(logit_diffs))
    max_prob_diff = float(max(prob_diffs))
    agreement_rate = float(agreements) / float(cfg.portability_benchmark_runs)
    degradation_pct = (1.0 - agreement_rate) * 100.0

    latency_ok = eager_mean <= cfg.max_latency_ms_cpu
    degradation_ok = (
        max_logit_diff <= cfg.max_export_degradation_tol
        and max_prob_diff <= cfg.max_export_degradation_tol
        and agreement_rate == 1.0
    )

    return PortabilityReport(
        parameter_count=param_count,
        parameter_memory_mb=round(param_memory_mb, 4),
        state_dict_size_mb=round(state_dict_size_mb, 4),
        torchscript_size_mb=round(torchscript_size_mb, 4),
        eager_latency_mean_ms=round(eager_mean, 3),
        eager_latency_std_ms=round(eager_std, 3),
        eager_latency_p95_ms=round(eager_p95, 3),
        torchscript_latency_mean_ms=round(ts_mean, 3),
        torchscript_latency_std_ms=round(ts_std, 3),
        end_to_end_latency_mean_ms=round(e2e_mean, 3),
        max_abs_logit_diff=max_logit_diff,
        max_abs_prob_diff=max_prob_diff,
        prediction_agreement_rate=agreement_rate,
        degradation_pct=degradation_pct,
        latency_within_sla=latency_ok,
        degradation_within_tol=degradation_ok,
        device=str(device),
        num_runs=cfg.portability_benchmark_runs,
    )
