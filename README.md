# Alzheimer ViT sMRI

Proyecto académico de clasificación futura de imágenes sMRI en CN, MCI y AD mediante Vision Transformers y entradas 2.5D.

**Autores:** Malak Sanchez y Juan Simancas. Universidad del Magdalena, Ingeniería de Sistemas, Seminario III, 2026-II.

## Estado del proyecto

- **E1 (S5):** Inventarios, criterios, particiones por sujeto y representación 2.5D. Aceptado (Hito H1).
- **E2 (S7):** Control de calidad, normalización parenquimal y extracción del triplete axial canónico 2.5D. Aceptado (Hito H2).
- **E3 (S9):** Arquitectura ViT-B/16 triclase (85.8M parámetros) implementada y validada. Aceptado (Hito H3).
- **E4 (S11):** Componentes de datos, entrenamiento, evaluación e inferencia integrada. Aceptado (Hito H4).
- **E5 (S12):** Módulo de interpretabilidad post-hoc mediante Attention Rollout. Aceptado (Hito H5).
- **Semana 13 (E6 en curso):** Integración del flujo extremo a extremo (E1/E2 -> E3 -> E4 -> E5) con datos sintéticos y evaluación del Indicador I9 de portabilidad TorchScript.
- **Reportes:** Versiones finales S5–S13 disponibles en [docs/reports](docs/reports/README.md).

## Instalación y ejecución

Desde la raíz del repositorio, con [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked
uv run --locked python scripts/run_e1_verification.py
uv run --locked pytest
uv run --locked python scripts/run_e2_demo.py --output-dir data/processed/mi_demo_e2
uv run --locked python scripts/run_s9_verification.py
```

Para generar los reportes desde su contenido estructurado:

```bash
uv run --locked --group reports python scripts/build_weekly_reports.py
```

El grupo opcional `reports` instala ReportLab. Esta orden actualiza los PDF finales en `docs/reports/`.

## Organización

```text
configs/    Configuración del preprocesamiento y contenido de reportes
data/       Inventarios y datos locales de ejecución
docs/       Línea base y reportes semanales
scripts/    Puntos de entrada para verificación, procesamiento y reportes
src/data/   Implementación de inventario, particiones y preprocesamiento
src/model/  Arquitectura ViT-B/16 triclase
src/training/ Componentes de entrenamiento futuro con verificación sintética
tests/      Pruebas de datos, arquitectura y entrenamiento futuro
```

## Documentación

- [Índice documental](docs/README.md)
- [Línea base](docs/base/Linea_Base_Proyecto_Alzheimer_ViT.pdf)
- [Implementación técnica S9](docs/implementation/E4_S9_Implementacion_ViT_y_Entrenamiento_Futuro.md)
- [Reportes semanales S5–S9](docs/reports/README.md)
- [Comandos y contrato de preprocesamiento](scripts/README.md)

La metodología conserva el triplete axial centrado en z = −12 mm en MNI152, la normalización independiente por corte y las particiones por sujeto. No se aprenden estadísticas de normalización entre particiones. Los inventarios originales son referencias de metadatos, no acreditación de acceso ni manifiestos listos para E2.
