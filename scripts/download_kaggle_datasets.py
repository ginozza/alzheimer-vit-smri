"""Download and extract Kaggle Alzheimer datasets for project experimentation.

Usage:
    python scripts/download_kaggle_datasets.py --token <KAGGLE_API_TOKEN>
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"

DATASETS = [
    {
        "name": "Kaggle Alzheimer 4-Class (ADNI/OASIS)",
        "slug": "preetpalsingh25/alzheimers-dataset-4-class-of-images",
        "target_dir": DATA_RAW / "kaggle_adni_4class",
    },
    {
        "name": "Kaggle Images OASIS (ninadaithal)",
        "slug": "ninadaithal/imagesoasis",
        "target_dir": DATA_RAW / "kaggle_oasis",
    },
]


def download_and_extract(slug: str, target_dir: Path, token: str, dataset_name: str):
    print("=" * 80)
    print(f"  DESCARGANDO: {dataset_name}")
    print(f"  Kaggle Slug: {slug}")
    print(f"  Destino:     {target_dir}")
    print("=" * 80)

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir.parent / f"{target_dir.name}_temp.zip"

    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")

    print("Conectando con la API de Kaggle...")
    start_time = time.time()
    with urllib.request.urlopen(req) as resp:
        total_size = resp.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else 0
        total_mb = total_bytes / (1024 * 1024)

        print(f"Tamano reportado: {total_mb:.2f} MB")
        print("Descargando archivo comprimido...")

        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB chunks

        with open(zip_path, "wb") as fp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Progreso: {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Descargados: {mb:.1f} MB", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDescarga finalizada en {elapsed:.1f} segundos. Descomprimiendo en {target_dir}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    print("Descompresion completada con exito.")
    # Remove temp zip to free disk space
    if zip_path.exists():
        zip_path.unlink()

    # Count extracted files
    files = list(target_dir.rglob("*"))
    img_files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Total de archivos extraidos: {len(files)} (Imagenes encontradas: {len(img_files)})")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets.")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("KAGGLE_API_TOKEN", ""),
        help="Kaggle API token (or set KAGGLE_API_TOKEN env var)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "adni", "oasis"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    token = args.token.strip()
    if not token:
        print("ERROR: Token de Kaggle no especificado.", file=sys.stderr)
        sys.exit(1)

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    to_download = []
    if args.dataset in ("all", "adni"):
        to_download.append(DATASETS[0])
    if args.dataset in ("all", "oasis"):
        to_download.append(DATASETS[1])

    for ds in to_download:
        download_and_extract(
            slug=ds["slug"],
            target_dir=ds["target_dir"],
            token=token,
            dataset_name=ds["name"],
        )

    print("TODAS LAS DESCARGAS Y EXTRACCIONES COMPLETADAS CON EXITO.")


if __name__ == "__main__":
    main()
