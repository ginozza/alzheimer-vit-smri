"""
All-in-one verification runner for Deliverable E1: Plan de Datos y Protocolo Experimental.
Executes data inventory audit, subject-level partitioning, and zero-leakage mathematical checks.

Usage:
    python run_e1_verification.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.inventory_loader import DatasetInventoryLoader
from src.data.subject_split import SubjectSplitter
from src.data.slice_representation import CanonicalTripletExtractor, Slice25DConfig


def main():
    print("=" * 80)
    print("  ENTREGABLE E1: PLAN DE DATOS Y PROTOCOLO EXPERIMENTAL - VERIFICACION")
    print("  Seminario III (2026-II) - Universidad del Magdalena")
    print("  Autores: Malak Sanchez, Juan Simancas | Director: Sergio Lubo")
    print("=" * 80)

    # 1. Verification of Inventories and Dictionaries
    print("\n[PASO 1] Cargando especificaciones de inventario y diccionarios de variables...")
    dict_file = Path("data/inventory/adni_oasis_data_dictionary.yaml")
    adni_file = Path("data/inventory/adni_metadata_inventory.csv")
    oasis_file = Path("data/inventory/oasis_metadata_inventory.csv")

    assert dict_file.exists(), f"ERROR: No se encontro {dict_file}"
    assert adni_file.exists(), f"ERROR: No se encontro {adni_file}"
    assert oasis_file.exists(), f"ERROR: No se encontro {oasis_file}"
    print(f"  [OK] Archivos de inventario encontrados y validados.")

    loader = DatasetInventoryLoader(dictionary_path=dict_file)
    combined_df, report = loader.load_combined_cohort([adni_file, oasis_file])

    print(f"  [OK] Total de escaneos cargados: {len(combined_df)}")
    print(f"  [OK] Sujetos unicos identificados: {combined_df['Subject_ID'].nunique()}")
    print(f"  [OK] Distribucion de clases: {report['combined']['class_distribution']}")

    # 2. Subject-level Partition (80% Dev / 20% Holdout)
    print("\n[PASO 2] Ejecutando particion estratificada a nivel de sujeto (Subject_ID)...")
    splitter = SubjectSplitter(random_seed=42)
    split_result = splitter.split_holdout(combined_df, test_size=0.20, val_size_from_dev=0.125)

    s_train = set(split_result.train_df["Subject_ID"])
    s_val = set(split_result.val_df["Subject_ID"])
    s_test = set(split_result.test_df["Subject_ID"])

    print(f"  [OK] Sujetos en Entrenamiento (Train):  {len(s_train)}")
    print(f"  [OK] Sujetos en Validacion (Val):        {len(s_val)}")
    print(f"  [OK] Sujetos en Prueba (Test Hold-out):  {len(s_test)}")

    # 3. Acceptance Criterion Verification: Zero Subject Leakage
    print("\n[PASO 3] Evaluando Criterio de Aceptacion de Linea Base:")
    print("  >> 'Ningun sujeto aparece en mas de una particion' <<")

    train_val_overlap = s_train.intersection(s_val)
    train_test_overlap = s_train.intersection(s_test)
    val_test_overlap = s_val.intersection(s_test)

    print(f"  - Solapamiento Train y Val:  {len(train_val_overlap)} sujetos")
    print(f"  - Solapamiento Train y Test: {len(train_test_overlap)} sujetos")
    print(f"  - Solapamiento Val y Test:   {len(val_test_overlap)} sujetos")

    assert len(train_val_overlap) == 0, "Fallo: Hay fuga entre Train y Val"
    assert len(train_test_overlap) == 0, "Fallo: Hay fuga entre Train y Test"
    assert len(val_test_overlap) == 0, "Fallo: Hay fuga entre Val y Test"
    print("  >>> VERIFICACION EXITOSA: CERO FUGA DE SUJETOS ENTRE PARTICIONES (Interseccion = Vacio) <<<")

    # 4. K-Fold Cross Validation Verification
    print("\n[PASO 4] Generando 5-Fold Cross Validation sobre el conjunto de desarrollo (80%)...")
    folds = splitter.generate_kfold_cv(split_result.development_df, n_splits=5)
    for idx, (f_train, f_val) in enumerate(folds):
        ft_s = set(f_train["Subject_ID"])
        fv_s = set(f_val["Subject_ID"])
        overlap = ft_s.intersection(fv_s)
        assert len(overlap) == 0, f"Fallo en Fold {idx}: Solapamiento de {len(overlap)} sujetos."
        print(f"  [OK] Fold {idx+1}/5: {len(ft_s)} sujetos en train, {len(fv_s)} en val (Solapamiento = 0)")

    # 5. 2.5D Tensor Dimension Verification
    print("\n[PASO 5] Verificando compatibilidad de dimensiones de entrada 2.5D para ViT-B/16...")
    extractor = CanonicalTripletExtractor()
    batch_x, batch_y = extractor.generate_synthetic_batch(batch_size=4)
    print(f"  [OK] Tensor sintetico 2.5D generado con forma: {batch_x.shape}")
    print(f"  [OK] Vector de etiquetas triclase generado con forma: {batch_y.shape}")

    assert batch_x.shape == (4, 3, 224, 224), f"Dimensiones inesperadas: {batch_x.shape}"
    assert set(batch_y).issubset({0, 1, 2}), f"Clases inesperadas: {batch_y}"
    print("  [OK] Dimensiones compatibles con ViT-B/16 (3 canales RGB, parches 16x16, 224x224 px).")

    print("\n" + "=" * 80)
    print("  CONCLUSION DE AUDITORIA: TODOS LOS CRITERIOS DE E1 HAN SIDO SUPERADOS CON EXITO.")
    print("=" * 80)


if __name__ == "__main__":
    main()
