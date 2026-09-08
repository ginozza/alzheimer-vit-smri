# Alzheimer ViT sMRI: Detección Prodrómica de la Enfermedad de Alzheimer mediante Vision Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)](#)
[![Seminario III](https://img.shields.io/badge/Universidad_del_Magdalena-Seminario_III_2026--II-darkblue.svg)](#)

Repositorio oficial del proyecto de investigación y desarrollo tecnológico para la futura clasificación triclase (**Control Normal - CN**, **Deterioro Cognitivo Leve - MCI** y **Enfermedad de Alzheimer - AD**) a partir de imágenes de Resonancia Magnética Estructural (sMRI), utilizando arquitecturas **Vision Transformer (ViT-B/16)** con entradas 2.5D e interpretabilidad clínica vía **Attention Rollout**.

* **Autores:** Malak Sanchez, Juan Simancas
* **Director:** Sergio Lubo
* **Institución:** Universidad del Magdalena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas
* **Periodo Académico:** Seminario III — 2026-II

---

## Estructura del Repositorio

```
alzheimer-vit-smri/
├── .gitignore                                 # Exclusiones de Git (entornos, datos pesados, pesos)
├── README.md                                  # Documentación principal del repositorio
├── requirements.txt                           # Dependencias científicas
├── pyproject.toml                             # Configuración estándar de empaquetado y pytest
├── run_e1_verification.py                     # Script ejecutable de auditoría y verificación integral de E1
│
├── data/
│   └── inventory/
│       ├── adni_oasis_data_dictionary.yaml    # Diccionario unificado de variables y esquema de metadatos
│       ├── adni_metadata_inventory.csv        # Manifiesto de inventario de la cohorte ADNI
│       ├── oasis_metadata_inventory.csv       # Manifiesto de inventario de la cohorte OASIS
│       └── cohort_summary.json                # Resumen estructurado de cohortes, fuentes y protocolos
│
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── inventory_loader.py                # Carga, validación de esquemas y filtros de inclusión/exclusión
│       ├── subject_split.py                   # Partición estratificada por sujeto (80/20 Holdout y 5-Fold CV)
│       └── slice_representation.py            # Extractor y modelado de entradas 2.5D (triplete canónico MNI152)
│
└── tests/
    ├── __init__.py
    ├── test_inventory.py                      # Pruebas de carga de inventario y filtros de exclusión
    ├── test_subject_leakage.py                # Verificación estricta de CERO fuga de sujetos (Criterio E1)
    └── test_slice_dimensions.py               # Verificación de dimensiones de tensor 2.5D [3, 224, 224]
```

---

## Entregable E1: Plan de Datos y Protocolo Experimental

El **Entregable 1 (E1)** responde al paquete **EDT 2.0 (Especificación de datos)** de la Línea Base:
- **Contenido:** Inventario, criterios, partición por sujeto, variables y representación 2.5D.
- **Evidencia:** Documento formal de la Línea Base, inventario (`data/inventory/`) y pruebas técnicas de separación por sujeto (`src/data/subject_split.py`, `tests/test_subject_leakage.py`).
- **Criterio de Aceptación:** *Los permisos y fuentes están registrados; las clases y exclusiones están definidas; ningún sujeto aparece en más de una partición.*

### Aspectos Clave Metodológicos:
1. **Fuentes:** ADNI (ADNI-1, ADNI-GO, ADNI-2, ADNI-3) y OASIS (OASIS-1, OASIS-3) con secuencias T1w MPRAGE.
2. **Clases Diagnósticas:** Triclase (CN, MCI, AD) parametrizadas por criterios clínicos estandarizados (CDR y MMSE).
3. **Representación 2.5D:** Triplete canónico axial de 3 cortes contiguos centrado en el hipocampo ($z = -12\text{ mm}$ en espacio MNI152), mapeados a canales RGB ($3 \times 224 \times 224$). Esta configuración ofrece la mayor uniformidad experimental y una reducción del $85\%$ en costo computacional frente a modelos 3D.
4. **Partición por Sujeto:** 80% Desarrollo (preparado para 5-fold cross validation estratificado) y 20% Test Hold-out reservado, con garantía matemática de cero fuga de información ($S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$).

---

## Instalación y Verificación Rápida

### 1. Clonar el repositorio y configurar el entorno virtual:
```bash
cd alzheimer-vit-smri
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux / macOS:
source .venv/bin/activate

# Instalar dependencias científicas:
pip install -r requirements.txt
```

### 2. Ejecutar la verificación automatizada del Entregable 1:
```bash
python run_e1_verification.py
```

### 3. Ejecutar la suite de pruebas unitarias con Pytest:
```bash
pytest tests/ -v
```

---

## Matriz de Cumplimiento de Criterios de Aceptación

| Criterio E1 | Archivo / Módulo de Verificación | Estado |
| :--- | :--- | :---: |
| Permisos y fuentes registrados | Línea Base (Secc. 2 y 3), `cohort_summary.json` | CUMPLIDO |
| Clases y exclusiones definidas | `data/inventory/adni_oasis_data_dictionary.yaml`, `src/data/inventory_loader.py` | CUMPLIDO |
| Ningún sujeto en más de una partición | `src/data/subject_split.py`, `tests/test_subject_leakage.py`, `run_e1_verification.py` | CUMPLIDO |
