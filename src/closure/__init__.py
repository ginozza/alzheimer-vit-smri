"""Subpackage for Week 15 & Deliverable E7: Project Closure, Release Candidate Audit & Traceability (EDT 8.0)."""

from .audit import (
    ReleaseCandidateAuditReport,
    RepeatabilityAuditResult,
    RepositoryHygieneAuditResult,
    TraceabilityAuditResult,
    audit_repository_hygiene,
    run_release_candidate_audit,
    verify_demo_repeatability,
    verify_project_traceability,
)
from .config import ClosureConfig

__all__ = [
    "ClosureConfig",
    "ReleaseCandidateAuditReport",
    "RepeatabilityAuditResult",
    "RepositoryHygieneAuditResult",
    "TraceabilityAuditResult",
    "audit_repository_hygiene",
    "run_release_candidate_audit",
    "verify_demo_repeatability",
    "verify_project_traceability",
]
