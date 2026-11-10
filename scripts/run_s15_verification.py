"""All-in-one formal verification runner for Week 15 & Deliverable E7: Cierre del proyecto (Versión candidata final).

Seminario III (2026-II) - Universidad del Magdalena
Equipo: Malak Sanchez, Juan Simancas | Director: Sergio Lubo
Semana 15 - Paquete EDT 8.0 (Cierre del proyecto / Entregable E7)

Audits the official acceptance criterion defined in the Linea Base (Seccion 6, pag. 4):
>> 'El informe mantiene trazabilidad con objetivos y resultados; el repositorio no contiene
>>  credenciales ni datos restringidos; la demostracion es repetible.' <<

Usage:
    uv run --locked python scripts/run_s15_verification.py
"""

from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.closure import (
    ClosureConfig,
    audit_repository_hygiene,
    run_release_candidate_audit,
    verify_demo_repeatability,
    verify_project_traceability,
)


def main() -> int:
    start_time = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]

    print("=" * 84)
    print("VERIFICACION FORMAL SEMANA 15 & ENTREGABLE E7: CIERRE Y VERSION CANDIDATA FINAL")
    print("=" * 84)

    # 1. Config & Module Audit
    print("\n[AUDITORIA 1] Verificando configuracion externa de cierre (configs/e7_closure.json)...")
    config = ClosureConfig.load(repo_root / "configs" / "e7_closure.json")
    assert config.release_candidate_version == "0.1.0-rc1"
    assert config.project_period == "2026-II"
    print(
        f"  [OK] Configuracion cargada: version={config.release_candidate_version} "
        f"(periodo={config.project_period})"
    )
    print("  [OK] Paquete EDT 8.0 | Entregable E7 (Semana 15)")

    # 2. Repository Hygiene & Security Audit (EDT 8.2)
    print("\n[AUDITORIA 2] Ejecutando depuracion e higiene del repositorio (EDT 8.2)...")
    hygiene = audit_repository_hygiene(repo_root, config)
    assert hygiene.passed is True, (
        f"Violaciones detectadas: {hygiene.credential_violations} / {hygiene.restricted_data_violations}"
    )
    assert hygiene.credential_violations == []
    assert hygiene.restricted_data_violations == []
    print(f"  [OK] Archivos inspeccionados: {hygiene.scanned_files_count}")
    print("  [OK] Cero credenciales expuestas (.env, API keys, tokens, claves privadas).")
    print("  [OK] Cero datos clinicos restringidos (.nii, .nii.gz, .dcm, .ima, .pem, .key).")

    # 3. Documentary Coverage (I10) & Traceability Matrix (EDT 8.1)
    print("\n[AUDITORIA 3] Verificando trazabilidad (OE1-OE4, E1-E7, I1-I10) y cobertura documental (I10)...")
    traceability = verify_project_traceability(repo_root, config)
    assert traceability.passed is True, f"Artefactos faltantes: {traceability.missing_artifacts}"
    assert traceability.documentary_coverage_pct == 100.0
    print(
        f"  [OK] Indicador I10 (Cobertura documental): "
        f"{traceability.verified_artifacts_count}/{traceability.total_required_artifacts} "
        f"({traceability.documentary_coverage_pct:.1f}% >= 90.0% requerido)"
    )
    print(
        "  [OK] Estado de entregables: "
        + ", ".join(f"{k}={v}" for k, v in traceability.deliverables_status.items())
    )

    # 4. Repeatable Demonstration Audit (EDT 8.3 & EDT 8.5)
    print("\n[AUDITORIA 4] Verificando repetibilidad determinista de la demostracion tecnica (EDT 8.5)...")
    repeatability = verify_demo_repeatability(seed=config.demo_seed, compact_model=False)
    assert repeatability.passed is True
    assert repeatability.max_logit_discrepancy == 0.0
    assert repeatability.max_probability_discrepancy == 0.0
    assert repeatability.untrained_notice_verified is True
    print(f"  [OK] Clases sinteticas evaluadas en doble corrida (seed={repeatability.seed}): {repeatability.tested_classes}")
    print(
        f"  [OK] Discrepancia numerica nula entre corridas: "
        f"max |d_logit|={repeatability.max_logit_discrepancy:.2e}, "
        f"max |d_prob|={repeatability.max_probability_discrepancy:.2e}"
    )
    print("  [OK] Advertencias de modelo no entrenado y no diagnostico verificadas en todas las corridas.")

    # 5. Full Release Candidate Report Export
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "e7_release_candidate_audit.json"
        full_report = run_release_candidate_audit(
            repo_root=repo_root,
            config=config,
            compact_demo=True,
        )
        full_report.to_json(out_json)
        assert full_report.overall_passed is True
        assert out_json.exists()

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 84)
    print(
        f"RESULTADO GLOBAL E7 / SEMANA 15: APROBADO ({elapsed:.2f} s) - "
        f"VERSION CANDIDATA FINAL {config.release_candidate_version}"
    )
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
