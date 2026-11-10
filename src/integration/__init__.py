"""Subpackage for Deliverable E6: System Integration, Technical Interface & Portability (S13-S14)."""

from .config import IntegrationConfig
from .interface import (
    TechnicalIntegrationInterface,
    build_cli_parser,
    run_cli,
)
from .pipeline import (
    IntegrationExecutionResult,
    IntegrationInputError,
    IntegratedSystemPipeline,
    UNTRAINED_MODEL_NOTICE,
    generate_synthetic_25d_case,
)
from .portability import (
    PortabilityReport,
    run_portability_benchmark,
)

__all__ = [
    "IntegrationConfig",
    "IntegrationExecutionResult",
    "IntegrationInputError",
    "IntegratedSystemPipeline",
    "PortabilityReport",
    "TechnicalIntegrationInterface",
    "UNTRAINED_MODEL_NOTICE",
    "build_cli_parser",
    "generate_synthetic_25d_case",
    "run_cli",
    "run_portability_benchmark",
]
