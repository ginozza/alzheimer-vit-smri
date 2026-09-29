# Ejecutables

Ejecutar desde la raíz del repositorio. La lógica reutilizable está en `src/data/`.

| Script | Función | Salida |
| --- | --- | --- |
| `run_e1_verification.py` | Verificar inventario, particiones y representación E1 | Resumen en terminal |
| `run_e2_demo.py` | Crear y comprobar ocho casos sintéticos E2 | Carpeta nueva con entradas, manifiesto, resultados y vista previa |
| `run_e2_preprocessing.py` | Procesar un manifiesto completo de un fold | Tensores y registros de QC en una carpeta nueva |
| `build_weekly_reports.py` | Componer reportes desde JSON | PDF finales en `docs/reports/` |

```bash
uv run --locked python scripts/run_e1_verification.py
uv run --locked python scripts/run_e2_demo.py --output-dir data/processed/mi_demo_e2
uv run --locked python scripts/run_e2_preprocessing.py --help
uv run --locked --group reports python scripts/build_weekly_reports.py
```

El demo retorna 0 cuando se cumplen los cuatro casos aceptados y cuatro rechazos previstos. El CLI de preprocesamiento retorna 0 sin rechazos, 1 con rechazos y 2 ante errores globales. Consultar el contrato de entrada siguiente antes de preparar un manifiesto.

## Contrato de entrada E2

El pipeline requiere NIfTI 3D T1w no negativos, con fondo cero, extracción cerebral previa y registro MNI152 documentado. No ejecuta registro, extracción cerebral ni conversión DICOM. Los CSV de `data/inventory/` no son manifiestos E2 completos.

| Campos del CSV | Valores requeridos |
| --- | --- |
| `Subject_ID`, `Session_ID`, `Cohort_Source` | Identidades no vacías, estables y sin espacios exteriores; sesión única por sujeto |
| `Split` | Train, Validation o Test; un sujeto pertenece a una sola partición |
| `Image_File_Path` | Ruta relativa a `--data-root`, sin salir de esa raíz |
| `Diagnosis_Class`, `Age`, `Sex`, `MMSE`, `CDR` | CN/MCI/AD; 55–95; M/F; 0–30; CDR en 0/0.5/1/2/3 |
| `Modality`, `Pulse_Sequence` | T1w_sMRI; MPRAGE, IR-FSPGR o T1-weighted 3D |
| `Space`, `Template_ID`, `Preprocessing_Provenance` | MNI152, plantilla exacta y descripción verificable de preparación previa |
| `Quality_Control_Pass`, `Brain_Extraction_Pass`, `Registration_QC_Pass` | true tras revisión real; en el demo la procedencia indica su carácter simulado |
| `Sample_Kind` | real o synthetic; nunca mezclados en una corrida |

Fuentes reales admitidas: ADNI-1/GO/2/3 y OASIS-1/3. Las pruebas sintéticas usan fuente SYNTHETIC e identificadores SYNTH_. Las etiquetas clínicas se validan estructuralmente, no se infieren diagnósticos.

Suministrar un manifiesto completo con train, validation y test de un mismo fold; conservar juntas todas las visitas de cada sujeto. La protección contra fuga depende de identidades correctas y del manifiesto completo suministrado. No compara corridas independientes ni detecta duplicados aproximados.

```bash
uv run --locked python scripts/run_e2_preprocessing.py \
  --manifest data/local_manifest.csv \
  --data-root data/raw \
  --config configs/e2.json \
  --output-dir data/processed/e2_run_001
```

## Transformaciones y registros

La configuración predeterminada exige voxeles de 1 mm (tolerancia 0.05 mm), orientación canónica RAS y geometría coherente. Se rechazan imágenes corruptas, no finitas, vacías, negativas, oblicuas, duplicados exactos canónicos y recortes que perderían señal. La conversión de ejes a RAS no es registro anatómico.

Se extraen tres cortes axiales contiguos centrados en z = −12 mm mediante el affine. Cada corte se normaliza sobre su señal positiva mediante clipping p1/p99 y z-score, sin estadísticas compartidas entre sujetos. El fondo permanece cero. Se aplica recorte/relleno centrado sin interpolación para obtener float32 de forma 3×224×224; no se aplica normalización ImageNet. Los canales van de inferior a superior, las filas corresponden a x RAS y las columnas a y RAS.

Cada carpeta de salida contiene `run.json` (configuración, versiones y hashes), `input_manifest.csv`, `qc.jsonl` (resultado por adquisición), `summary.json`, `outputs.csv` y tensores NPY separados por partición. Los rechazos incluyen un motivo estable; una corrida incompleta o con errores no debe consumirse como conjunto válido. Los directorios de salida existentes se rechazan.

Para revisar una muestra reproducible, ejecutar el demo: crea `raw/`, `manifest.csv`, `processed/` y `preview.png`. La vista compara entrada y salida; los registros permiten comprobar su procedencia. Los ejemplos son sintéticos y no acreditan calidad anatómica, aceptación académica ni rendimiento diagnóstico. Los datos reales ADNI/OASIS siguen pendientes.
