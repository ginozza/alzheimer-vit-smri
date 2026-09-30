"""Quick validation of E2 preprocessing pipeline on real Kaggle dataset samples.

Processes a small subset (up to 5 images per class) from each dataset to
verify end-to-end correctness without needing to preprocess all ~99k images.

Usage:
    uv run python scripts/validate_real_datasets.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.kaggle_adapter import (
    KaggleAdapterConfig,
    KaggleDatasetScanner,
    normalize_class_name,
    preprocess_2d_slice,
)


DATASETS = [
    {
        "name": "KAGGLE-ADNI-4CLASS",
        "root": ROOT / "data" / "raw" / "kaggle_adni_4class",
    },
    {
        "name": "KAGGLE-OASIS",
        "root": ROOT / "data" / "raw" / "kaggle_oasis",
    },
]

MAX_PER_CLASS = 5


def validate_dataset(name: str, root: Path) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  Validando: {name}")
    print(f"  Ruta:      {root}")
    print(f"{'=' * 70}")

    if not root.exists():
        print(f"  ADVERTENCIA: directorio no encontrado, omitiendo.")
        return {"dataset": name, "status": "SKIPPED", "reason": "directory_not_found"}

    # 1. Scan all class folders
    scanner = KaggleDatasetScanner(root, cohort_name=name)
    records = scanner.scan()
    print(f"  Total de registros escaneados: {len(records)}")

    if not records:
        print(f"  ERROR: No se encontraron imagenes validas.")
        return {"dataset": name, "status": "FAIL", "reason": "no_images_found"}

    # 2. Compute class distribution
    class_counts = {}
    for r in records:
        cls = r["Diagnosis_Class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  Distribucion triclase: {class_counts}")

    # 3. Process a small sample per class
    config = KaggleAdapterConfig()
    sampled = {}
    passed = 0
    failed = 0

    for r in records:
        cls = r["Diagnosis_Class"]
        if cls not in sampled:
            sampled[cls] = 0
        if sampled[cls] >= MAX_PER_CLASS:
            continue

        img_path = root / r["Image_File_Path"]
        try:
            tensor = preprocess_2d_slice(img_path, config)
            assert tensor.shape == (3, 224, 224), f"Shape inesperada: {tensor.shape}"
            assert tensor.dtype.name == "float32", f"Dtype inesperado: {tensor.dtype}"
            passed += 1
        except Exception as exc:
            print(f"  FALLO [{cls}] {r['Image_File_Path']}: {exc}")
            failed += 1

        sampled[cls] += 1

    total_sampled = sum(sampled.values())
    print(f"  Imagenes muestreadas: {total_sampled}")
    print(f"  Aprobadas:            {passed}")
    print(f"  Rechazadas:           {failed}")

    status = "PASS" if failed == 0 and passed > 0 else "FAIL"
    print(f"  Resultado: {status}")

    return {
        "dataset": name,
        "status": status,
        "total_scanned": len(records),
        "class_distribution": class_counts,
        "sampled": total_sampled,
        "passed": passed,
        "failed": failed,
    }


def main():
    print("=" * 70)
    print("  VALIDACION E2: PIPELINE DE PREPROCESAMIENTO CON DATOS REALES")
    print("=" * 70)

    results = []
    for ds in DATASETS:
        result = validate_dataset(ds["name"], ds["root"])
        results.append(result)

    print(f"\n{'=' * 70}")
    print("  RESUMEN FINAL")
    print(f"{'=' * 70}")
    all_pass = True
    for r in results:
        marker = "OK" if r["status"] == "PASS" else "FAIL"
        print(f"  [{marker}] {r['dataset']}: {r.get('total_scanned', 'N/A')} imagenes, "
              f"{r.get('passed', 0)} aprobadas, {r.get('failed', 0)} rechazadas")
        if r["status"] != "PASS":
            all_pass = False

    if all_pass:
        print("\n  VALIDACION COMPLETA: Ambos datasets procesados exitosamente.")
    else:
        print("\n  ADVERTENCIA: Algunos datasets no pasaron la validacion.")
        sys.exit(1)

    # Save validation report
    report_path = ROOT / "data" / "processed" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\n  Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
