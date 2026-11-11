"""Local technical integration interface (EDT 7.2 / Deliverable E6 - Semana 14).

Provides the user-facing programmatic and CLI technical interface required by
Deliverable E6 acceptance criteria:
- Receives a test input (synthetic 2.5D MRI case, .npy tensor file, or in-memory tensor).
- Executes the simulated end-to-end technical flow (E2 -> E3 -> E4 -> E5).
- Controls and reports errors gracefully with structured error codes without unhandled crashes.
- Explicitly states that the system does NOT contain a trained model and does NOT
  generate clinical diagnoses.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from src.interpretability.config import RolloutConfig
from src.model.config import ViTConfig
from .config import IntegrationConfig
from .pipeline import (
    IntegrationExecutionResult,
    IntegratedSystemPipeline,
    UNTRAINED_MODEL_NOTICE,
)
from .portability import PortabilityReport, run_portability_benchmark


class TechnicalIntegrationInterface:
    """Local technical interface connecting test inputs to the simulated ViT-B/16 flow."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        vit_config_path: Optional[Union[str, Path]] = None,
        rollout_config_path: Optional[Union[str, Path]] = None,
    ):
        self.config = (
            IntegrationConfig.load(config_path)
            if config_path is not None
            else IntegrationConfig()
        )
        self.vit_config = (
            ViTConfig.load(vit_config_path)
            if vit_config_path is not None
            else ViTConfig(
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
                in_channels=self.config.in_channels,
                num_classes=self.config.num_classes,
                class_names=self.config.class_names,
            )
        )
        self.rollout_config = (
            RolloutConfig.load(rollout_config_path)
            if rollout_config_path is not None
            else RolloutConfig(
                image_size=self.config.image_size,
                patch_size=self.config.patch_size,
            )
        )
        self.pipeline = IntegratedSystemPipeline(
            integration_config=self.config,
            vit_config=self.vit_config,
            rollout_config=self.rollout_config,
        )

    def execute_test_input(
        self,
        input_data: Optional[Union[str, Path, np.ndarray, torch.Tensor]] = None,
        synthetic_class: str = "MCI",
        seed: Optional[int] = None,
        subject_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> IntegrationExecutionResult:
        """Execute the simulated technical flow on a provided input or synthetic test sample.

        Never raises unhandled input exceptions: all contract violations are captured
        and returned inside IntegrationExecutionResult with status='ERROR'.
        """
        if input_data is None:
            return self.pipeline.run_synthetic_case(
                target_class=synthetic_class,
                seed=seed,
                subject_id=subject_id,
                output_dir=output_dir,
                safe_mode=True,
            )

        if isinstance(input_data, (str, Path)):
            return self.pipeline.run_on_npy_file(
                file_path=input_data,
                subject_id=subject_id,
                output_dir=output_dir,
                safe_mode=True,
            )

        resolved_subject = subject_id or "TEST_INPUT_TENSOR"
        return self.pipeline.run_on_tensor(
            x=input_data,
            subject_id=resolved_subject,
            input_source="tensor",
            output_dir=output_dir,
            safe_mode=True,
        )

    def execute_portability_check(
        self,
        export_dir: Optional[Union[str, Path]] = None,
    ) -> PortabilityReport:
        """Execute Indicator I9 portability benchmark from the technical interface."""
        return run_portability_benchmark(
            model=self.pipeline.model,
            integration_config=self.config,
            rollout_config=self.rollout_config,
            export_dir=export_dir,
        )

    @staticmethod
    def format_execution_report(result: IntegrationExecutionResult) -> str:
        """Format an IntegrationExecutionResult into a clear console report."""
        lines = [
            "=" * 80,
            "  INTERFAZ TECNICA DE INTEGRACION - PROTOTIPO E6 (ALZHEIMER ViT-B/16 sMRI)",
            "=" * 80,
            f"  Estado de ejecucion:       {result.status}",
            f"  Identificador de sujeto:   {result.subject_id}",
            f"  Fuente de entrada:         {result.input_source}",
            f"  Modelo entrenado:          {'SI' if result.is_trained_model else 'NO (Pesos no entrenados - Flujo tecnico simulado)'}",
            f"  Genera diagnostico clinico:{'SI' if result.generates_clinical_diagnosis else 'NO (Exclusivamente verificacion de software)'}",
            "-" * 80,
        ]

        if result.status == "SUCCESS":
            lines.extend([
                f"  Dimensiones de entrada:    {result.input_shape} ({result.input_dtype})",
                f"  Clase simulada de origen:  {result.simulated_target_class or 'N/A (Archivo/Tensor externo)'}",
                f"  Salida tecnica triclase:   {result.predicted_class_name} (ID={result.predicted_class_id}, Confianza={result.confidence:.4f})",
                f"  Probabilidades Softmax:    {result.probabilities}",
                f"  Logits crudos (1x3):       {result.logits}",
                f"  Attention Rollout Grid:    {result.cls_grid_shape} -> Heatmap {result.heatmap_shape} rango {result.heatmap_range}",
                f"  Latencia del flujo:        {result.latency_ms:.2f} ms",
            ])
            if result.saved_artifacts:
                lines.append("  Artefactos exportados:")
                for k, v in result.saved_artifacts.items():
                    lines.append(f"    - {k}: {v}")
        else:
            lines.extend([
                f"  Codigo de error controlado:{result.error_code}",
                f"  Detalle del error:         {result.error_message}",
                f"  Latencia de validacion:    {result.latency_ms:.2f} ms",
            ])

        lines.extend([
            "-" * 80,
            f"  {result.untrained_model_notice}",
            "-" * 80,
            f"  {result.clinical_disclaimer}",
            "=" * 80,
        ])
        return "\n".join(lines)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interfaz tecnica local de integracion (Entregable E6) para el flujo simulado "
            "2.5D sMRI -> ViT-B/16 -> Attention Rollout. No contiene un modelo entrenado "
            "ni genera diagnosticos clinicos."
        )
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Ruta a un archivo tensor .npy de entrada con forma (3, 224, 224) float32.",
    )
    parser.add_argument(
        "--synthetic-class",
        type=str,
        default="MCI",
        help="Clase sintetica a simular cuando no se provee --input-file (CN, MCI o AD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista para generacion de muestra sintetica (default: 42).",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Identificador opcional del sujeto de prueba.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio opcional donde exportar JSON, tensor .npy, heatmap.png y overlay.png.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/e6_integration.json",
        help="Ruta al archivo de configuracion de integracion E6.",
    )
    return parser


def run_cli(argv: Optional[List[str]] = None) -> Tuple[int, IntegrationExecutionResult]:
    """Execute the technical interface CLI and return (exit_code, result)."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    if cfg_path is not None and not cfg_path.is_file():
        cfg_path = None

    interface = TechnicalIntegrationInterface(config_path=cfg_path)
    result = interface.execute_test_input(
        input_data=args.input_file,
        synthetic_class=args.synthetic_class,
        seed=args.seed,
        subject_id=args.subject_id,
        output_dir=args.output_dir,
    )
    print(interface.format_execution_report(result))
    exit_code = 0 if result.status == "SUCCESS" else 1
    return exit_code, result
