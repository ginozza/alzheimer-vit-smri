# Informe de Avance Semanal — Semana 13 (S13)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural (`alzheimer-vit-smri`)  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena — Programa de Ingeniería de Sistemas (Seminario III, 2026-II)  
**Periodo:** Semana 13 | 26 de octubre a 01 de noviembre de 2026  
**Paquete EDT:** EDT 7.0 — Integración (Componentes 7.1 Flujo y 7.3 Datos sintéticos) + Indicador I9 (Portabilidad)  
**Entregable asociado:** Entregable E6: Prototipo de integración (en desarrollo hacia Hito H6 en S14)  
**Estado general:** Verde  
**PDF oficial:** [`Reporte_semanal_S13_final.pdf`](Reporte_semanal_S13_final.pdf)

---

## 1. Objetivos

De acuerdo con la planificación oficial establecida en la **Línea Base del Proyecto** (`docs/base/Linea_Base_Proyecto_Alzheimer_ViT.pdf`, Secciones 7, 8, 11 y 14):

1. **Objetivo del Cronograma (Semana 13):** *«Desarrollo e integración con el flujo completo utilizando datos sintéticos»*, obteniendo como evidencia un **«Flujo integrado funcional»** bajo responsabilidad conjunta de ambos integrantes.
2. **Componentes EDT 7.0 a desarrollar en S13:**
   - **EDT 7.1 (Flujo):** Conectar de extremo a extremo la representación 2.5D y validación de entrada (E1/E2), la arquitectura `VisionTransformer` ViT-B/16 triclase (E3), el pipeline de inferencia diagnóstica (E4) y el motor de interpretabilidad `AttentionRollout` con superposición visual (E5).
   - **EDT 7.3 (Datos sintéticos):** Implementar un generador determinista de casos sMRI 2.5D $(3, 224, 224)$ `float32` para las tres categorías clínicas (`CN`, `MCI`, `AD`), compatible con el extractor canónico de tripletes axiales.
3. **Objetivo de Seguimiento (Indicador I9 — Portabilidad):** Ejecutar y documentar en la Semana 13 la medición de **Tamaño, Latencia y Degradación** del modelo ViT-B/16 mediante empaquetamiento y exportación reproducible.

---

## 2. Contexto

Al iniciar la Semana 13, el proyecto registraba un avance acumulado del **71,43 % en entregables aceptados (5 de 7)** y el cumplimiento del 100 % de los hitos vencidos (H1 a H5):

- **Entregable E1 (S5 — H1):** Plan de datos, inventario clínico, particionado estratificado por sujeto (`SubjectSplitter`) sin fuga de datos ($S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$) y extractor canónico 2.5D (`CanonicalTripletExtractor`).
- **Entregable E2 (S7 — H2):** Control de calidad NIfTI 3D y adaptador de cortes axiales 2D de Kaggle (OASIS/ADNI), normalización por corte ($p_1/p_{99}$ + z-score) y tensores $(3, 224, 224)$ `float32`.
- **Entregable E3 (S8 — H3):** Arquitectura `VisionTransformer` (ViT-B/16 triclase `CN`/`MCI`/`AD`) con **85.800.963 parámetros entrenables** y retención desacoplada de mapas de atención en las 12 capas codificadoras.
- **Entregable E4 (S11 — H4):** Código fuente integrado de carga de datos (`MRI25DDataset`, `SyntheticMRIDataset`, `create_mri_dataloader`), entrenamiento futuro (`FutureTrainer`, `TrainingHistory`, pérdida ponderada y AdamW), evaluación multiclase (`EvaluationMetrics`, matriz $3 \times 3$) e inferencia (`InferencePipeline`, `PredictionResult`).
- **Entregable E5 (S12 — H5):** Módulo de interpretabilidad `AttentionRollout` (Abnar & Zuidema, 2020), colormap pseudo-Jet, superposición alfa sobre cortes axiales y descargo clínico obligatorio (`CLINICAL_DISCLAIMER`), cerrando con 157 pruebas unitarias aprobadas.

---

## 3. Actividades realizadas

1. **Diseño y externalización de la configuración de integración (`configs/e6_integration.json` y `src/integration/config.py`):**
   - Creación de `IntegrationConfig` como `dataclass(frozen=True)` con validación de hiperparámetros geométricos ($(3, 224, 224)$, parche $16 \times 16$), clases objetivo `("CN", "MCI", "AD")`, semilla determinista (`42`), banderas de salvaguarda académica (`is_trained_model=False`, `enable_clinical_diagnosis=False`) y parámetros del benchmark de portabilidad I9.
2. **Implementación del generador sintético 2.5D determinista (`generate_synthetic_25d_case` en `src/integration/pipeline.py`):**
   - Construcción de volúmenes elipsoidales sintéticos 3D ($128 \times 128 \times 128$) con modulación de cavidad ventricular/hipocampal dependiente de la clase simulada (`CN`, `MCI`, `AD`) y extracción de tres cortes axiales contiguos mediante `CanonicalTripletExtractor` y `Slice25DConfig` (E1/E2).
3. **Desarrollo del orquestador de flujo completo (`IntegratedSystemPipeline` en `src/integration/pipeline.py`):**
   - Encadenamiento directo de validación de contrato tensorial (`validate_input_array`) $\rightarrow$ `InferencePipeline.predict_tensor` (E4) $\rightarrow$ `AttentionRollout.explain_tensor` (E5) $\rightarrow$ renderizado visual con `overlay_heatmap_on_slice`, `save_heatmap_png` y `save_overlay_png`.
   - Implementación de `run_full_synthetic_cohort` para procesar cohortes balanceadas y generar el resumen auditable `cohort_integration_summary.json`.
4. **Implementación del módulo de portabilidad e Indicador I9 (`src/integration/portability.py`):**
   - Desarrollo de `run_portability_benchmark` y `PortabilityReport` para cuantificar el tamaño en RAM y disco (`state_dict` y `TorchScript`), la latencia de inferencia en CPU (PyTorch Eager, TorchScript y flujo extremo a extremo) y la degradación numérica tras la exportación mediante `torch.jit.trace`.
5. **Creación del verificador ejecutable y documentación técnica S13:**
   - Implementación de `scripts/run_s13_verification.py`, `tests/test_s13_integration_and_portability.py` y `docs/implementation/E6_S13_Flujo_Integrado_y_Portabilidad.md`.

---

## 4. Implementación

| Archivo / Módulo | Tipo | Descripción Técnica |
|---|---|---|
| [`configs/e6_integration.json`](../../configs/e6_integration.json) | Creado | Parámetros externos desacoplados del flujo integrado E6 y umbrales de portabilidad I9. |
| [`src/integration/config.py`](../../src/integration/config.py) | Creado | Dataclass `IntegrationConfig` con validación en `__post_init__` y serialización JSON. |
| [`src/integration/pipeline.py`](../../src/integration/pipeline.py) | Creado | `IntegratedSystemPipeline`, `IntegrationExecutionResult`, `IntegrationInputError`, `UNTRAINED_MODEL_NOTICE` y `generate_synthetic_25d_case`. |
| [`src/integration/portability.py`](../../src/integration/portability.py) | Creado | `PortabilityReport` y `run_portability_benchmark` para el Indicador I9 (Tamaño, latencia y degradación). |
| [`src/integration/__init__.py`](../../src/integration/__init__.py) | Creado | Exportación limpia de la API pública del subpaquete `src.integration`. |
| [`scripts/run_s13_verification.py`](../../scripts/run_s13_verification.py) | Creado | Script de auditoría reproducible para el flujo integrado S13 y el Indicador I9. |
| [`tests/test_s13_integration_and_portability.py`](../../tests/test_s13_integration_and_portability.py) | Creado | 10 pruebas unitarias y de integración para S13. |
| [`docs/implementation/E6_S13_Flujo_Integrado_y_Portabilidad.md`](../implementation/E6_S13_Flujo_Integrado_y_Portabilidad.md) | Creado | Especificación técnica del flujo integrado S13 y benchmark de portabilidad. |

---

## 5. Decisiones técnicas

1. **Reutilización estricta de `CanonicalTripletExtractor` (`Slice25DConfig`) en el generador sintético E6:** En lugar de crear un generador de tensores aleatorios desconectado de E1/E2, `generate_synthetic_25d_case` sintetiza un volumen cerebral 3D con fondo cero y aplica el extractor canónico oficial de E1/E2 (clipping $p_1/p_{99}$ + z-score sobre señal positiva y recorte centrado a $224 \times 224$), garantizando fidelidad con el contrato de preprocesamiento.
2. **Evaluación de portabilidad (I9) mediante trazado `TorchScript` (`torch.jit.trace`) con envoltorio de logits:** Dado que el alcance semestral excluye el entrenamiento de pesos sobre datos reales, la degradación de portabilidad se definió rigurosamente como la discrepancia numérica máxima ($\max |\Delta \text{logits}|$ y $\max |\Delta \text{probs}|$) y la tasa de acuerdo de predicción entre el modelo PyTorch Eager y su grafo serializado en TorchScript.
3. **Inmutabilidad de banderas de restricción académica:** `IntegrationConfig` rechaza en tiempo de instanciación cualquier intento de activar `enable_clinical_diagnosis=True`, asegurando por diseño el cumplimiento de las exclusiones de la Línea Base.

---

## 6. Resultados

La ejecución del runner `uv run --locked python scripts/run_s13_verification.py` arrojó los siguientes resultados verificables:

| Métrica / Dimensión Evaluada | Resultado Verificado |
|---|---|
| **Cohorte sintética integrada (EDT 7.1 / 7.3)** | 6 sujetos sintéticos (`2 CN`, `2 MCI`, `2 AD`) procesados de extremo a extremo con 100 % de éxito (`status="SUCCESS"`). |
| **Artefactos por sujeto** | Generación verificada de `input_25d.npy` $(3, 224, 224)$, `attention_heatmap.png` $(224, 224)$, `attention_overlay.png` $(224, 224, 3)$ y `integration_result.json`. |
| **Indicador I9 — Tamaño (Parámetros / RAM)** | **85.800.963 parámetros** entrenables (**327,30 MB** en memoria `float32`). |
| **Indicador I9 — Tamaño en disco (`state_dict` / `TorchScript`)** | Checkpoint `.pt`: **327,36 MB** \| Grafo exportado TorchScript `.pt`: **327,58 MB**. |
| **Indicador I9 — Latencia en CPU (B=1)** | PyTorch Eager: **600,76 ms $\pm$ 59,90 ms** ($p_{95} = 676,49\text{ ms}$) \| TorchScript: **585,54 ms $\pm$ 49,41 ms** \| Flujo E2E con Attention Rollout: **638,59 ms**. |
| **Indicador I9 — Degradación de exportación** | **0,00 % de degradación** (concordancia de clases **100,0 %**; $\max |\Delta \text{logit}| = 0,00$; $\max |\Delta \text{prob}| = 0,00$). |

---

## 7. Validaciones

1. **Verificador de Semana 13 (`scripts/run_s13_verification.py`):** Ejecutado y aprobado al 100 % en las 4 auditorías (configuración externa, generación sintética determinista, flujo integrado sobre cohorte de 6 casos y benchmark de portabilidad I9).
2. **Pruebas automatizadas (`tests/test_s13_integration_and_portability.py`):** **10/10 pruebas aprobadas**.
3. **Suite acumulada al cierre de S13:** **167/167 pruebas unitarias e integración aprobadas**.

---

## 8. Dificultades

1. **Desfase del puntero de la rama `dev` respecto a `main`:** La rama local `dev` permanecía en el commit de la Semana 9 (`d2660ca`), mientras que las Semanas 10, 11 y 12 estaban integradas en `main` (`fd3ad59`). Se resolvió adelantando `dev` para incorporar la historia completa sin bifurcaciones.
2. **Nomenclatura de configuración en `src/data/slice_representation.py`:** Al integrar el generador sintético con el extractor de E1, se verificó que la clase de configuración se denomina `Slice25DConfig` y el método `extract_triplet_from_volume`, asegurando compatibilidad estricta.
3. **Advertencias de trazado en PyTorch 2.x durante `torch.jit.trace`:** Se encapsuló el trazado con un contexto controlado de advertencias y se verificó mediante pruebas numéricas que el grafo exportado reproduce exactamente los logits del modelo Eager ($\Delta = 0,00$).

---

## 9. Limitaciones

1. **Ausencia de pesos entrenados (Restricción de Línea Base):** El flujo integrado opera con pesos inicializados estocásticamente conforme a la restricción académica de Seminario III; las probabilidades y mapas de atención verifican el flujo computacional y no poseen validez diagnóstica clínica.
2. **Latencia medida exclusivamente en CPU:** Al no contar con GPU dedicada en el entorno de validación local, las mediciones del Indicador I9 corresponden a ejecución en CPU en precisión simple (`float32`).

---

## 10. Conclusiones

Se cumplieron al **100 %** los objetivos programados para la **Semana 13**: el flujo completo del sistema (`E1/E2 -> E3 -> E4 -> E5`) se encuentra integrado y funcional utilizando datos sintéticos deterministas, y el **Indicador I9 (Portabilidad)** quedó formalmente medido y documentado con **0,00 % de degradación** entre el modelo PyTorch y su exportación TorchScript.

---

## 11. Trabajo futuro

Para la **Semana 14 (S14)** se programan las actividades de cierre del **Entregable E6 (Hito H6)**:
1. Implementar la interfaz técnica local (`TechnicalIntegrationInterface` y CLI) con control exhaustivo de errores de entrada.
2. Incorporar y auditar los avisos obligatorios de modelo no entrenado y ausencia de diagnóstico clínico en todas las salidas de la interfaz.
3. Elaborar el **Manual de Uso e Instalación** del sistema (`EDT 7.4`) y ejecutar la auditoría formal de aceptación de E6 (`scripts/run_e6_verification.py`).
