"""CLI runner to execute E2 preprocessing on Kaggle 2D MRI datasets.

Usage:
    uv run python scripts/run_e2_kaggle_pipeline.py --input-dir data/raw/kaggle_oasis --output-dir data/processed/e2_kaggle_oasis
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.kaggle_adapter import KaggleAdapterConfig, execute_kaggle_preprocessing_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Kaggle 2D MRI axial slices into ViT-B/16 compatible tensors."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to folder containing Kaggle dataset images organized by class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory for manifests, QC logs, and NPY tensors.",
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="KAGGLE-OASIS",
        help="Cohort identifier (e.g., KAGGLE-OASIS, KAGGLE-ADNI-4CLASS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject-level stratified partitioning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("  ENTREGABLE E2: PREPROCESAMIENTO DE CONJUNTOS DE DATOS KAGGLE")
    print(f"  Cohorte: {args.cohort_name}")
    print(f"  Entrada: {args.input_dir}")
    print(f"  Salida:  {args.output_dir}")
    print("=" * 80)

    if not args.input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {args.input_dir}", file=sys.stderr)
        sys.exit(2)

    config = KaggleAdapterConfig(random_seed=args.seed)

    try:
        summary = execute_kaggle_preprocessing_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cohort_name=args.cohort_name,
            config=config,
        )
    except Exception as exc:
        print(f"ERROR DURANTE LA EJECUCION: {exc}", file=sys.stderr)
        sys.exit(2)

    print("\n[RESULTADOS DEL PROCESAMIENTO]")
    print(f"  Total de imagenes analizadas: {summary['Total_Images_Found']}")
    print(f"  Casos aprobados por QC:       {summary['Processed_Passed']}")
    print(f"  Casos rechazados:             {summary['Rejected']}")
    print(f"  Distribucion triclase:        {summary['Class_Distribution']}")
    print(f"  Distribucion de particion:    {summary['Split_Distribution']}")
    print(f"  Dimensiones del tensor final: {summary['Tensor_Shape']} ({summary['Tensor_Dtype']})")
    print("\n  Artefactos generados exitosamente en:")
    print(f"  - {args.output_dir / 'input_manifest.csv'}")
    print(f"  - {args.output_dir / 'outputs.csv'}")
    print(f"  - {args.output_dir / 'qc.jsonl'}")
    print(f"  - {args.output_dir / 'summary.json'}")
    print(f"  - {args.output_dir / 'tensors/'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
