"""End-to-end integrated system pipeline for Deliverable E6 (S13-S14).

Connects all project subsystems across Deliverables E1 through E5:
- E1/E2: Canonical 2.5D axial triplet representation (3, 224, 224) float32,
         synthetic MRI slice generation, and strict input contract validation.
- E3:    VisionTransformer (ViT-B/16 triclass, 85,800,963 parameters).
- E4:    InferencePipeline (triclass CN/MCI/AD logits and Softmax probabilities).
- E5:    AttentionRollout (Abnar & Zuidema 2020) and pseudo-Jet visual overlay.
- E6:    Unified execution flow, structured error handling, artifact export,
         and mandatory untrained-model / non-diagnostic notices.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data.slice_representation import CanonicalTripletExtractor, Slice25DConfig
from src.inference.predictor import CLASS_NAMES, InferencePipeline
from src.interpretability.config import RolloutConfig
from src.interpretability.rollout import (
    CLINICAL_DISCLAIMER,
    AttentionRollout,
)
from src.interpretability.visualization import (
    overlay_heatmap_on_slice,
    save_heatmap_png,
    save_overlay_png,
)
from src.model.config import ViTConfig
from src.model.vit import VisionTransformer
from .config import IntegrationConfig


UNTRAINED_MODEL_NOTICE: str = (
    "AVISO OBLIGATORIO DE PROTOTIPO TECNICO (ENTREGABLE E6 / LINEA BASE): "
    "Esta interfaz ejecuta un flujo tecnico simulado con datos de prueba y NO contiene "
    "un modelo entrenado ni genera diagnosticos clinicos. Los pesos del modelo ViT-B/16 "
    "corresponden a inicializacion no entrenada conforme al alcance academico del "
    "periodo 2026-II; las probabilidades y mapas generados verifican exclusivamente la "
    "integridad dimensional, funcional y computacional del sistema."
)


class IntegrationInputError(ValueError):
    """Structured error raised when an input violates the E6 integration contract."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class IntegrationExecutionResult:
    """Structured output returned by the E6 integration pipeline and technical interface."""

    status: str
    subject_id: str
    input_source: str
    input_shape: Optional[Tuple[int, ...]] = None
    input_dtype: Optional[str] = None
    simulated_target_class: Optional[str] = None
    predicted_class_id: Optional[int] = None
    predicted_class_name: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    logits: Optional[List[float]] = None
    heatmap_shape: Optional[Tuple[int, int]] = None
    cls_grid_shape: Optional[Tuple[int, int]] = None
    heatmap_range: Optional[Tuple[float, float]] = None
    saved_artifacts: Dict[str, str] = field(default_factory=dict)
    is_trained_model: bool = False
    generates_clinical_diagnosis: bool = False
    untrained_model_notice: str = UNTRAINED_MODEL_NOTICE
    clinical_disclaimer: str = CLINICAL_DISCLAIMER
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.input_shape is not None:
            data["input_shape"] = list(self.input_shape)
        if self.heatmap_shape is not None:
            data["heatmap_shape"] = list(self.heatmap_shape)
        if self.cls_grid_shape is not None:
            data["cls_grid_shape"] = list(self.cls_grid_shape)
        if self.heatmap_range is not None:
            data["heatmap_range"] = list(self.heatmap_range)
        return data

    def to_json(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
        return target


def generate_synthetic_25d_case(
    target_class: str = "MCI",
    seed: int = 42,
    image_size: int = 224,
) -> np.ndarray:
    """Generate a deterministic, normalized 2.5D sMRI axial triplet (3, 224, 224) float32.

    Uses CanonicalTripletExtractor (E1/E2) to extract three contiguous axial slices
    centered at z = -12 mm MNI152 from a synthetic brain ellipsoid volume modulated
    by class-specific ventricular/hippocampal intensity signatures.

    Args:
        target_class: Diagnostic class label to simulate ("CN", "MCI", or "AD").
        seed: Deterministic random seed.
        image_size: Target spatial resolution (default 224).

    Returns:
        NumPy array of shape (3, image_size, image_size) and dtype float32.
    """
    if target_class not in CLASS_NAMES:
        raise IntegrationInputError(
            "INVALID_TARGET_CLASS",
            f"Clase sintetica '{target_class}' invalida. Clases permitidas: {CLASS_NAMES}.",
        )

    rng = np.random.default_rng(seed)
    vol_shape = (128, 128, 128)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, vol_shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, vol_shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, vol_shape[2], dtype=np.float32),
        indexing="ij",
    )

    # Ellipsoidal brain mask + anatomical intensity variation
    radius_sq = (zz / 0.80) ** 2 + (yy / 0.85) ** 2 + (xx / 0.75) ** 2
    brain_mask = radius_sq <= 1.0

    class_idx = CLASS_NAMES.index(target_class)
    # Subtle synthetic ventricular/hippocampal cavity modulation by class
    cavity_radius = 0.16 + 0.05 * class_idx
    cavity_mask = (zz ** 2 + (yy / 1.2) ** 2 + (xx / 0.9) ** 2) <= (cavity_radius ** 2)

    base_signal = np.where(brain_mask, 100.0 * (1.0 - 0.35 * radius_sq), 0.0).astype(np.float32)
    base_signal = np.where(cavity_mask & brain_mask, base_signal * (0.45 - 0.10 * class_idx), base_signal)
    noise = rng.normal(loc=0.0, scale=4.0, size=vol_shape).astype(np.float32)
    volume = np.clip(base_signal + np.where(brain_mask, noise, 0.0), 0.0, None).astype(np.float32)

    extractor = CanonicalTripletExtractor(
        Slice25DConfig(
            canonical_z_index=vol_shape[2] // 2,
            target_height=image_size,
            target_width=image_size,
            channels=3,
        )
    )
    triplet = extractor.extract_triplet_from_volume(volume)
    return triplet.astype(np.float32)


class IntegratedSystemPipeline:
    """Orchestrates the complete E1 -> E2 -> E3 -> E4 -> E5 flow for Deliverable E6."""

    def __init__(
        self,
        integration_config: Optional[IntegrationConfig] = None,
        vit_config: Optional[ViTConfig] = None,
        rollout_config: Optional[RolloutConfig] = None,
        model: Optional[VisionTransformer] = None,
    ):
        self.config = integration_config or IntegrationConfig()
        self.vit_config = vit_config or ViTConfig(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            num_classes=self.config.num_classes,
            class_names=self.config.class_names,
        )
        self.rollout_config = rollout_config or RolloutConfig(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
        )

        torch.manual_seed(self.config.default_seed)
        self.device = torch.device(self.config.default_device)
        self.model = (model if model is not None else VisionTransformer(self.vit_config)).to(self.device)
        self.model.eval()

        self.inference_pipeline = InferencePipeline(
            model=self.model,
            device=self.device,
            class_names=self.config.class_names,
        )
        self.rollout_engine = AttentionRollout(
            model=self.model,
            config=self.rollout_config,
            device=self.device,
        )

    def validate_input_array(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, Tuple[int, ...], str]:
        """Validate a 2.5D MRI input tensor against the E2 -> E6 contract.

        Raises:
            IntegrationInputError: With a stable error_code if any check fails.
        """
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            dtype_str = str(x.dtype).replace("torch.", "")
        elif isinstance(x, np.ndarray):
            arr = x
            dtype_str = str(x.dtype)
        else:
            raise IntegrationInputError(
                "INVALID_INPUT_TYPE",
                f"Se esperaba np.ndarray o torch.Tensor, pero se recibio {type(x).__name__}.",
            )

        orig_shape = tuple(int(d) for d in arr.shape)

        if arr.size == 0:
            raise IntegrationInputError(
                "EMPTY_INPUT_TENSOR",
                f"El tensor de entrada esta vacio (shape={orig_shape}).",
            )

        if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
            raise IntegrationInputError(
                "INVALID_DTYPE",
                f"Tipo de dato no numerico o no soportado: {dtype_str}.",
            )

        arr_f32 = arr.astype(np.float32)

        if not np.all(np.isfinite(arr_f32)):
            raise IntegrationInputError(
                "NON_FINITE_VALUES",
                "El tensor de entrada contiene valores no finitos (NaN o Inf).",
            )

        expected_3d = (self.config.in_channels, self.config.image_size, self.config.image_size)
        if arr_f32.ndim == 4:
            if arr_f32.shape[0] != 1:
                raise IntegrationInputError(
                    "UNSUPPORTED_BATCH_SIZE",
                    f"La interfaz unitaria requiere lote B=1 o tensor 3D {expected_3d}; se recibio {orig_shape}.",
                )
            arr_f32 = arr_f32[0]
        elif arr_f32.ndim != 3:
            raise IntegrationInputError(
                "INVALID_DIMENSIONS",
                f"Rango tensorial invalido ({arr_f32.ndim}D). Se requiere un triplete 2.5D con forma {expected_3d}.",
            )

        if tuple(arr_f32.shape) != expected_3d:
            raise IntegrationInputError(
                "SHAPE_MISMATCH",
                f"Dimensiones incompatibles: se recibio {tuple(arr_f32.shape)}, se esperaba {expected_3d}.",
            )

        if float(np.std(arr_f32)) < 1e-8:
            raise IntegrationInputError(
                "DEGENERATE_CONSTANT_SIGNAL",
                "El corte de entrada tiene varianza nula (senal constante o vacia) y no supera el control de calidad.",
            )

        return arr_f32, orig_shape, dtype_str

    def run_on_tensor(
        self,
        x: Union[np.ndarray, torch.Tensor],
        subject_id: str = "SYNTH_SUBJ_001",
        input_source: str = "tensor",
        simulated_target_class: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        safe_mode: bool = True,
    ) -> IntegrationExecutionResult:
        """Execute the full integrated flow on an in-memory tensor or array."""
        t0 = time.perf_counter()
        try:
            arr_f32, orig_shape, dtype_str = self.validate_input_array(x)

            # 1. Inference (E4 + E3)
            pred_result = self.inference_pipeline.predict_tensor(arr_f32, return_attention=True)

            # 2. Interpretability via Attention Rollout (E5)
            rollout_result = self.rollout_engine.explain_tensor(arr_f32)
            heatmap_2d = rollout_result.heatmap[0].detach().cpu().numpy()
            cls_grid_2d = rollout_result.cls_attention_grid[0].detach().cpu().numpy()

            # Verify overlay synthesis (E5 visualization contract)
            _ = overlay_heatmap_on_slice(arr_f32[1], heatmap_2d, alpha=self.config.default_alpha)

            saved_artifacts: Dict[str, str] = {}
            if output_dir is not None:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)

                tensor_path = out_path / f"{subject_id}_input_25d.npy"
                np.save(tensor_path, arr_f32)

                heatmap_path = save_heatmap_png(
                    heatmap_2d, out_path / f"{subject_id}_attention_heatmap.png"
                )
                overlay_path = save_overlay_png(
                    arr_f32[1],
                    heatmap_2d,
                    out_path / f"{subject_id}_attention_overlay.png",
                    alpha=self.config.default_alpha,
                )
                saved_artifacts["input_tensor_npy"] = str(tensor_path)
                saved_artifacts["heatmap_png"] = str(heatmap_path)
                saved_artifacts["overlay_png"] = str(overlay_path)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result = IntegrationExecutionResult(
                status="SUCCESS",
                subject_id=subject_id,
                input_source=input_source,
                input_shape=orig_shape,
                input_dtype=dtype_str,
                simulated_target_class=simulated_target_class,
                predicted_class_id=pred_result.predicted_class_id,
                predicted_class_name=pred_result.predicted_class_name,
                confidence=round(pred_result.confidence, 6),
                probabilities={k: round(v, 6) for k, v in pred_result.probabilities.items()},
                logits=[round(val, 6) for val in pred_result.logits],
                heatmap_shape=(int(heatmap_2d.shape[0]), int(heatmap_2d.shape[1])),
                cls_grid_shape=(int(cls_grid_2d.shape[0]), int(cls_grid_2d.shape[1])),
                heatmap_range=(round(float(np.min(heatmap_2d)), 6), round(float(np.max(heatmap_2d)), 6)),
                saved_artifacts=saved_artifacts,
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=None,
                error_message=None,
                latency_ms=round(elapsed_ms, 3),
            )

            if output_dir is not None:
                json_path = Path(output_dir) / f"{subject_id}_integration_result.json"
                result.to_json(json_path)
                saved_artifacts["result_json"] = str(json_path)

            return result

        except IntegrationInputError as exc:
            if not safe_mode:
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=subject_id,
                input_source=input_source,
                simulated_target_class=simulated_target_class,
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=exc.error_code,
                error_message=exc.message,
                latency_ms=round(elapsed_ms, 3),
            )
        except Exception as exc:
            if not safe_mode:
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=subject_id,
                input_source=input_source,
                simulated_target_class=simulated_target_class,
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code="UNEXPECTED_PIPELINE_ERROR",
                error_message=str(exc),
                latency_ms=round(elapsed_ms, 3),
            )

    def run_on_npy_file(
        self,
        file_path: Union[str, Path],
        subject_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        safe_mode: bool = True,
    ) -> IntegrationExecutionResult:
        """Load a 2.5D .npy tensor file from disk and execute the integrated flow."""
        t0 = time.perf_counter()
        path = Path(file_path)
        resolved_subject = subject_id or path.stem

        if not path.exists() or not path.is_file():
            err = IntegrationInputError(
                "FILE_NOT_FOUND",
                f"El archivo de entrada no existe o no es un archivo regular: {path}",
            )
            if not safe_mode:
                raise err
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=resolved_subject,
                input_source="npy_file",
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=err.error_code,
                error_message=err.message,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        if path.suffix.lower() != ".npy":
            err = IntegrationInputError(
                "UNSUPPORTED_FILE_EXTENSION",
                f"Formato de archivo no soportado ('{path.suffix}'). Se requiere un tensor NumPy '.npy'.",
            )
            if not safe_mode:
                raise err
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=resolved_subject,
                input_source="npy_file",
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=err.error_code,
                error_message=err.message,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:
            err = IntegrationInputError(
                "CORRUPT_NPY_FILE",
                f"No fue posible decodificar el archivo .npy '{path}': {exc}",
            )
            if not safe_mode:
                raise err
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=resolved_subject,
                input_source="npy_file",
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=err.error_code,
                error_message=err.message,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        return self.run_on_tensor(
            arr,
            subject_id=resolved_subject,
            input_source="npy_file",
            output_dir=output_dir,
            safe_mode=safe_mode,
        )

    def run_synthetic_case(
        self,
        target_class: str = "MCI",
        seed: Optional[int] = None,
        subject_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        safe_mode: bool = True,
    ) -> IntegrationExecutionResult:
        """Generate a synthetic 2.5D MRI case and execute the full integrated flow."""
        t0 = time.perf_counter()
        actual_seed = self.config.default_seed if seed is None else seed
        resolved_subject = subject_id or f"SYNTH_{target_class}_{actual_seed:03d}"

        try:
            synthetic_tensor = generate_synthetic_25d_case(
                target_class=target_class,
                seed=actual_seed,
                image_size=self.config.image_size,
            )
        except IntegrationInputError as exc:
            if not safe_mode:
                raise
            return IntegrationExecutionResult(
                status="ERROR",
                subject_id=resolved_subject,
                input_source="synthetic",
                simulated_target_class=target_class,
                is_trained_model=False,
                generates_clinical_diagnosis=False,
                untrained_model_notice=UNTRAINED_MODEL_NOTICE,
                clinical_disclaimer=CLINICAL_DISCLAIMER,
                error_code=exc.error_code,
                error_message=exc.message,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        return self.run_on_tensor(
            synthetic_tensor,
            subject_id=resolved_subject,
            input_source="synthetic",
            simulated_target_class=target_class,
            output_dir=output_dir,
            safe_mode=safe_mode,
        )

    def run_full_synthetic_cohort(
        self,
        samples_per_class: int = 2,
        seed: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[IntegrationExecutionResult]:
        """Execute the integrated flow across a balanced synthetic cohort (CN, MCI, AD)."""
        if samples_per_class < 1:
            raise ValueError("samples_per_class must be >= 1")

        base_seed = self.config.default_seed if seed is None else seed
        results: List[IntegrationExecutionResult] = []
        sample_counter = 0

        for cls_name in self.config.class_names:
            for idx in range(samples_per_class):
                case_seed = base_seed + sample_counter * 17
                subj_id = f"SYNTH_{cls_name}_{idx + 1:02d}"
                sub_dir = (Path(output_dir) / subj_id) if output_dir is not None else None
                res = self.run_synthetic_case(
                    target_class=cls_name,
                    seed=case_seed,
                    subject_id=subj_id,
                    output_dir=sub_dir,
                    safe_mode=True,
                )
                results.append(res)
                sample_counter += 1

        if output_dir is not None:
            summary_path = Path(output_dir) / "cohort_integration_summary.json"
            summary_payload = {
                "total_cases": len(results),
                "successful_cases": sum(1 for r in results if r.status == "SUCCESS"),
                "failed_cases": sum(1 for r in results if r.status == "ERROR"),
                "is_trained_model": False,
                "generates_clinical_diagnosis": False,
                "untrained_model_notice": UNTRAINED_MODEL_NOTICE,
                "clinical_disclaimer": CLINICAL_DISCLAIMER,
                "cases": [r.to_dict() for r in results],
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as fp:
                json.dump(summary_payload, fp, indent=2, ensure_ascii=False)

        return results
