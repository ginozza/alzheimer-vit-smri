# Entregable E7: Estructura de Presentación y Guion de Sustentación Final
## Material de Sustentación Académica (EDT 8.3 y 8.5 — Semanas 15 y 16)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Curso:** Seminario III (2026-II) — Programa de Ingeniería de Sistemas, Universidad del Magdalena

---

## 1. Estructura de Diapositivas para la Sustentación (14 Láminas)

### Diapositiva 1 — Portada Institucional
- **Título:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural.
- **Autores:** Malak Sánchez y Juan Simancas.
- **Director:** Ing. Sergio Lubo.
- **Programa:** Ingeniería de Sistemas — Seminario III (2026-II), Universidad del Magdalena.

### Diapositiva 2 — Planteamiento del Problema y Motivación Clínica
- El diagnóstico clínico de la enfermedad de Alzheimer (`AD`) suele ocurrir en etapas avanzadas con daño neuronal irreversible.
- La detección en la fase prodrómica —**Deterioro Cognitivo Leve (`MCI`)**— mediante resonancia magnética estructural Ponderada en T1 (`sMRI`) es crítica para habilitar intervenciones tempranas.
- **Reto computacional:** Los modelos 3D densos exigen alto consumo de memoria $\mathcal{O}(N^2)$ en mecanismos de atención y sufren sobreajuste en cohortes médicas limitadas.

### Diapositiva 3 — Alcance Semestral y Restricciones (Línea Base 2026-II)
- **Objetivo General:** Diseñar, implementar y documentar la arquitectura y el código fuente de un sistema basado en ViT-B/16 para la futura clasificación triclase (`CN`, `MCI`, `AD`) sin incluir el entrenamiento definitivo del modelo por restricciones de cómputo.
- **Exclusiones explícitas:** Entrenamiento en GPU de pesos clínicos, reporte de métricas diagnósticas sobre pacientes reales y uso asistencial hospitalario.

### Diapositiva 4 — Entregable E1: Plan de Datos y Particionado sin Fuga (Semana 5 — Hito H1)
- Especificación de cohortes **ADNI** y **OASIS** bajo acuerdos DUA (`data/inventory/`).
- Algoritmo `SubjectSplitter` con particionado estratificado a nivel de paciente:
  $$S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$$
- Evita que múltiples sesiones longitudinales de un mismo sujeto contaminen validación o prueba.

### Diapositiva 5 — Entregable E2: Preprocesamiento y Representación 2.5D (Semanas 6–7 — Hito H2)
- Extracción de **triplete axial canónico** centrado en el hipocampo ($z = -12\text{ mm}$ en espacio MNI152).
- Control de calidad geométrico RAS, recorte percentil $p_1/p_{99}$ y normalización z-score por corte sobre señal positiva (fondo conservado en cero).
- Adaptador para neuroimágenes reales de Kaggle (OASIS y ADNI 4-Class: 99.237 imágenes validadas) hacia tensores uniformes $(3, 224, 224)$ `float32`.
- **Ventaja:** Reducción del **85 % en costo computacional** y superior al **99 % en memoria de auto-atención** frente a 3D denso.

### Diapositiva 6 — Entregable E3: Arquitectura ViT-B/16 Triclase (Semana 8 — Hito H3)
- Adaptación de **ViT-B/16** (Dosovitskiy et al., 2020) con formulación **Pre-LayerNorm**:
  - `PatchEmbedding`: Convolución $16 \times 16 \to 196$ parches $+ 1$ token `[CLS]` $= 197$ tokens de dimensión $D = 768$ ($742.656$ parámetros).
  - `Encoder`: 12 bloques Transformer, 12 cabezas de atención, $\text{MLP} = 3072$ ($85.054.464$ parámetros).
  - `ClassificationHead`: `LayerNorm` + Proyección lineal a 3 clases `[CN, MCI, AD]` ($3.843$ parámetros).
  - **Total verificado:** **85.800.963 parámetros entrenables**.

### Diapositiva 7 — Entregable E4: Código Fuente de Entrenamiento, Evaluación e Inferencia (Semanas 9–11 — Hito H4)
- **Carga de datos:** `MRI25DDataset`, `SyntheticMRIDataset` y `create_mri_dataloader`.
- **Entrenamiento futuro:** Pérdida de entropía cruzada ponderada por frecuencia inversa ($w_c = N / (3 N_c)$ calculada solo en `train`), optimizador `AdamW` con exclusión selectiva de *weight decay*, recorte de gradiente y `FutureTrainer` con persistencia de checkpoints `.pt`.
- **Evaluación e inferencia:** Matriz de confusión $3 \times 3$, sensibilidad, especificidad y F1 macro en `EvaluationMetrics`, y pipeline `InferencePipeline`.

### Diapositiva 8 — Entregable E5: Interpretabilidad mediante Attention Rollout (Semana 12 — Hito H5)
- Formulación recursiva de Abnar & Zuidema (2020) sobre las 12 capas del codificador:
  $$\hat{A}_l = 0.5 \bar{A}_l + 0.5 I, \quad R_l = \hat{A}_l R_{l-1}$$
- Extracción de atención espacial del token `[CLS]` a cuadrícula $14 \times 14$, interpolación bilineal a $224 \times 224$, colormap pseudo-Jet y fusión $\alpha$-blending sobre el corte axial.
- Mitigación del **Riesgo R7**: inclusión obligatoria de `CLINICAL_DISCLAIMER`.

### Diapositiva 9 — Semana 13: Flujo Integrado e Indicador I9 de Portabilidad
- Tubería unificada `IntegratedSystemPipeline` conectando E1/E2 $\to$ E3 $\to$ E4 $\to$ E5.
- **Resultado del Indicador I9 (Portabilidad):**
  - **Tamaño:** 327,30 MB en RAM (`float32`), 327,36 MB en checkpoint `.pt`, 327,58 MB en `TorchScript`.
  - **Latencia en CPU:** ~585–601 ms por muestra individual ($B=1$).
  - **Degradación:** **0,00 %** (100 % de concordancia y $\Delta_{\max} = 0,00$ entre PyTorch Eager y TorchScript).

### Diapositiva 10 — Entregable E6: Interfaz Técnica y Control de Errores (Semana 14 — Hito H6)
- `TechnicalIntegrationInterface` y CLI ejecutable con soporte para entradas sintéticas, archivos `.npy` y tensores en memoria.
- Control determinista de **11 categorías de error** (`FILE_NOT_FOUND`, `CORRUPT_NPY_FILE`, `SHAPE_MISMATCH`, `NON_FINITE_VALUES`, `DEGENERATE_CONSTANT_SIGNAL`, etc.).
- Declaración explícita en consola y JSON (`UNTRAINED_MODEL_NOTICE`): **no contiene un modelo entrenado ni genera diagnósticos clínicos**.

### Diapositiva 11 — Entregable E7: Higiene de Repositorio, Trazabilidad y Pruebas (Semanas 15–16 — Hito H7)
- Suite acumulada de **193 pruebas automatizadas aprobadas al 100 %** en 15 módulos de prueba.
- Auditoría automatizada de cierre (`src/closure/audit.py`):
  - **0 credenciales o claves filtradas** y **0 datos clínicos restringidos**.
  - **100 % de cobertura documental (Indicador I10: 23/23 artefactos obligatorios)**.
  - **100 % de repetibilidad determinista** en la demostración técnica.

### Diapositiva 12 — Demostración Técnica en Vivo (Guion Paso a Paso)
1. Ejecución de `uv run --locked python scripts/run_e6_verification.py`: muestra el flujo simulado, los artefactos exportados (`heatmap.png`, `overlay.png`, `result.json`), el manejo de errores y los descargos clínicos.
2. Ejecución de `uv run --locked python scripts/run_s15_verification.py`: acredita en vivo la ausencia de credenciales, la trazabilidad completa y la repetibilidad.

### Diapositiva 13 — Cumplimiento de Indicadores de la Línea Base (I1 a I10)
- Presentación de la tabla consolidada de indicadores: 100 % de actividades cumplidas (I1), 100 % de hitos vencidos en fecha (I3), 0 riesgos altos abiertos (I5), 100 % de reproducibilidad (I7), portabilidad documentada con 0 % de degradación (I9) y 100 % de cobertura documental (I10).

### Diapositiva 14 — Conclusiones y Trabajo Futuro (Fase Post-Seminario III)
- Arquitectura y código fuente listos para entrenamiento en GPU sobre cohortes completas ADNI/OASIS siguiendo el protocolo de 5 pasos documentado en el Manual Técnico E6.
- Validación radiológica futura de mapas de Attention Rollout frente a segmentaciones de hipocampo y corteza entorrinal.
