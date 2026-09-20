# Alzheimer ViT sMRI

Proyecto académico de clasificación futura de imágenes sMRI en CN, MCI y AD mediante Vision Transformers y entradas 2.5D.

**Autores:** Malak Sanchez y Juan Simancas. Universidad del Magdalena, Ingeniería de Sistemas, Seminario III, 2026-II.

## Estado del proyecto

- **E1:** inventarios, criterios, particiones por sujeto y representación 2.5D. Las pruebas verifican separación sobre los manifiestos suministrados; el loader no aplica todo el esquema clínico.
- **E2:** control de calidad NIfTI, normalización por corte, generación de tensores float32 de 3×224×224 y registros reproducibles. Validado con muestras sintéticas.
- **Reportes:** versiones finales S5, S6 y S7 disponibles en [docs/reports](docs/reports/README.md).

## Instalación y ejecución

Desde la raíz del repositorio, con [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked
uv run --locked python scripts/run_e1_verification.py
uv run --locked pytest
uv run --locked python scripts/run_e2_demo.py --output-dir data/processed/mi_demo_e2
```

Para generar los tres reportes desde su contenido estructurado:

```bash
uv run --locked --group reports python scripts/build_weekly_reports.py
```

El grupo opcional `reports` instala ReportLab. Esta orden actualiza los PDF finales en `docs/reports/`;

## Organización

```text
configs/    Configuración del preprocesamiento y contenido de reportes
data/       Inventarios y datos locales de ejecución
docs/       Línea base y reportes semanales
scripts/    Puntos de entrada para verificación, procesamiento y reportes
src/data/   Implementación de inventario, particiones y preprocesamiento
tests/      Pruebas de E1, E2 y prevención de fuga de datos
```

## Documentación

- [Índice documental](docs/README.md)
- [Línea base](docs/base/Linea_Base_Proyecto_Alzheimer_ViT.pdf)
- [Reportes semanales S5–S7](docs/reports/README.md)
- [Comandos y contrato de preprocesamiento](scripts/README.md)

La metodología conserva el triplete axial centrado en z = −12 mm en MNI152, la normalización independiente por corte y las particiones por sujeto. No se aprenden estadísticas de normalización entre particiones. Los inventarios originales son referencias de metadatos, no acreditación de acceso ni manifiestos listos para E2.
