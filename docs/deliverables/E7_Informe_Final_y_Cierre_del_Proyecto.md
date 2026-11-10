# Entregable E7: Informe Final y Consolidación de Cierre del Proyecto
## Documento Técnico de Cierre Definitivo y Sustentación (Semana 16 — Hito H7)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas  
**Período Académico:** Seminario III (2026-II)  
**Fecha de corte:** Semana 16 (16 a 22 de noviembre de 2026)  
**Versión del sistema:** `alzheimer-vit-smri==0.1.0` (Versión Definitiva Final)  
**Estado:** **Formalmente Aceptado al 100 % (Hito H7 Cumplido — Cierre Oficial de Seminario III)**

---

## 1. Resumen Ejecutivo

Durante el periodo académico **2026-II (Semanas 5 a 16)** se diseñó, implementó, verificó y documentó la arquitectura completa y el código fuente de un sistema modular de apoyo investigativo basado en **Vision Transformers (ViT-B/16)** para la futura clasificación triclase de resonancia magnética estructural (sMRI) en **Control Normal (`CN`, 0)**, **Deterioro Cognitivo Leve (`MCI`, 1 — etapa prodrómica)** y **Enfermedad de Alzheimer (`AD`, 2)**.

En estricto cumplimiento de las restricciones de alcance y capacidad de cómputo definidas en la **Línea Base del Proyecto**, el sistema opera sobre una representación **2.5D de tres cortes axiales contiguos centrados en el hipocampo ($z = -12\text{ mm}$ en el espacio MNI152)**, reduciendo en más de un **85 % el costo computacional** frente a arquitecturas 3D densas, garantizando **cero fuga de datos por sujeto** ($S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$) y proveyendo interpretabilidad post-hoc mediante **Attention Rollout** e interfaz técnica local con avisos explícitos de modelo no entrenado y ausencia de diagnóstico clínico.

---

## 2. Evaluación de Criterios de Aceptación del Entregable E7 (Línea Base, Pág. 4)

> *«El informe mantiene trazabilidad con objetivos y resultados; el repositorio no contiene credenciales ni datos restringidos; la demostración es repetible.»*

| Criterio de Aceptación E7 | Evidencia Implementada y Auditada (`scripts/run_s16_verification.py`) | Dictamen |
|---|---|---|
| **Trazabilidad con objetivos y resultados** | Matriz completa OE1–OE7 $\leftrightarrow$ E1–E7 (Sección 3) y verificación automatizada (`verify_project_traceability`) con **100 % de cobertura documental (Indicador I10)**. | **SUPERADO** |
| **Ausencia de credenciales y datos restringidos** | Escaneo automatizado (`audit_repository_hygiene`) sobre la totalidad de archivos del repositorio: **0 credenciales hardcodeadas** y **0 volúmenes clínicos restringidos por DUA**. | **SUPERADO** |
| **Demostración repetible** | Verificación determinista (`verify_demo_repeatability`, `seed=42`) sobre las 3 clases (`CN`, `MCI`, `AD`) con discrepancia numérica nula ($\Delta = 0,00$). | **SUPERADO** |

---

## 3. Matriz de Trazabilidad General: Objetivos Específicos, Entregables y Resultados (EDT 8.1)

| Objetivo Específico (Línea Base 4.1) | Entregable / Hito | Paquete EDT | Módulos y Configuraciones | Pruebas y Evidencia Cuantitativa | Estado |
|---|---|---|---|---|---|
| **OE1:** Definir la estructura de datos, criterios de selección y reglas de separación por sujeto. | **E1 (H1 — S5)** | EDT 2.0 (2.1–2.5) | `src/data/inventory_loader.py`, `src/data/subject_split.py`, `data/inventory/` | `test_inventory.py`, `test_subject_leakage.py`, `run_e1_verification.py` ($S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$) | **ACEPTADO (100 %)** |
| **OE2:** Diseñar y codificar el proceso de preparación y construcción de entradas sMRI en formato 2.5D. | **E2 (H2 — S7)** | EDT 3.0 (3.1–3.5) | `src/data/slice_representation.py`, `src/data/preprocessing.py`, `src/data/kaggle_adapter.py`, `configs/e2.json` | `test_preprocessing.py`, `test_kaggle_adapter.py`, 99.237 imágenes Kaggle validadas, tensores $(3, 224, 224)$ `float32` | **ACEPTADO (100 %)** |
| **OE3:** Definir la arquitectura de software y la adaptación de ViT-B/16 para tres clases. | **E3 (H3 — S8)** | EDT 4.0 (4.1–4.5) | `src/model/config.py`, `src/model/layers.py`, `src/model/vit.py`, `configs/e3_architecture.json` | `test_vit_architecture.py` (31 pruebas), **85.800.963 parámetros**, `E3_Especificacion_Arquitectura_ViT.md` | **ACEPTADO (100 %)** |
| **OE4:** Implementar los módulos de carga de datos, modelo, entrenamiento futuro, evaluación e inferencia. | **E4 (H4 — S11)** | EDT 5.0 (5.1–5.5) | `src/data/dataset.py`, `src/training/`, `src/inference/predictor.py`, `configs/e4_*.json` | `test_dataset.py`, `test_training.py`, `test_trainer.py`, `test_evaluation.py`, `test_inference.py`, `test_interoperability.py` | **ACEPTADO (100 %)** |
| **OE5:** Implementar el módulo de Attention Rollout para su utilización posterior con pesos entrenados. | **E5 (H5 — S12)** | EDT 6.0 (6.1–6.4) | `src/interpretability/config.py`, `rollout.py`, `visualization.py`, `configs/e5_interpretabilidad.json` | `test_attention_rollout.py` (13 pruebas), `run_e5_verification.py`, `CLINICAL_DISCLAIMER` (Riesgo R7) | **ACEPTADO (100 %)** |
| **OE6:** Verificar mediante pruebas unitarias las dimensiones, interfaces y flujo de los componentes. | **E1–E7 (S5–S16)** | Transversal (3.5, 5.1–5.5, 6.3, 7.2, 8.2) | `tests/` (16 suites de prueba), `src/integration/portability.py`, `src/closure/audit.py` | **200/200 pruebas automatizadas aprobadas al 100 %**; Indicador I9 (327,30 MB, 0,00 % degradación) | **ACEPTADO (100 %)** |
| **OE7:** Documentar la instalación, la configuración y el procedimiento de entrenamiento futuro. | **E6 (H6 — S14) y E7 (S15–S16)** | EDT 7.0 y EDT 8.0 | `src/integration/`, `src/closure/`, `configs/e6_integration.json`, `configs/e7_closure.json` | `E6_Prototipo_Integracion_y_Manual_de_Uso.md`, `E7_Informe_Final_y_Cierre_del_Proyecto.md`, `E7_Presentacion_y_Sustentacion.md` | **ACEPTADO (100 %)** |

---

## 4. Consolidación Técnica de Subpaquetes del Sistema (`alzheimer-vit-smri==0.1.0`)

El repositorio final depurado está estructurado en **7 subpaquetes altamente cohesivos y desacoplados** bajo `src/`:

1. **`src.data` (EDT 2.0, 3.0 y 5.1):**
   - `inventory_loader.py` y `subject_split.py`: Ingesta de metadatos ADNI/OASIS y particionado estratificado por paciente con invariante matemática de cero fuga.
   - `slice_representation.py` y `preprocessing.py`: Control de calidad geométrico RAS, extracción de triplete axial canónico en $z = -12\text{ mm}$, recorte percentil $p_1/p_{99}$ y normalización z-score sobre señal positiva.
   - `kaggle_adapter.py`: Adaptador taxonómico de 4 clases a estándar triclase (`CN`, `MCI`, `AD`).
   - `dataset.py` y `dataloader.py`: `MRI25DDataset`, `SyntheticMRIDataset` y `create_mri_dataloader`.
2. **`src.model` (EDT 4.0 y 5.2):**
   - `ViTConfig`, `PatchEmbedding` ($16 \times 16 \to 196$ parches + `[CLS]`), 12 bloques `TransformerBlock` Pre-Norm ($D=768, H=12, \text{MLP}=3072$), `ClassificationHead` triclase y `VisionTransformer` con captura desacoplada de pesos de atención $(B, 12, 197, 197)$.
3. **`src.training` (EDT 5.3 y 5.4):**
   - `TrainingConfig`, `balanced_class_weights`, `build_classification_loss`, `build_optimizer` (AdamW con decaimiento selectivo), `train_batch`, `FutureTrainer`, `TrainingHistory` y `EvaluationMetrics` (matriz $3 \times 3$, precisión, sensibilidad, especificidad y F1 macro).
4. **`src.inference` (EDT 5.5):**
   - `InferencePipeline` y `PredictionResult` con inferencia unitaria, por lote y desde archivo `.npy`.
5. **`src.interpretability` (EDT 6.0):**
   - `RolloutConfig`, `AttentionRollout` ($R_l = \hat{A}_l R_{l-1}$ con conexión residual $(A+I)/2$), `RolloutResult`, `CLINICAL_DISCLAIMER`, `pseudo_jet_colormap` y `overlay_heatmap_on_slice`.
6. **`src.integration` (EDT 7.0):**
   - `IntegrationConfig`, `IntegratedSystemPipeline`, `TechnicalIntegrationInterface`, CLI `run_cli`, `UNTRAINED_MODEL_NOTICE` y benchmark del Indicador I9 `run_portability_benchmark`.
7. **`src.closure` (EDT 8.0):**
   - `ClosureConfig`, `audit_repository_hygiene`, `verify_project_traceability`, `verify_demo_repeatability` y `run_release_candidate_audit`.

---

## 5. Depuración del Repositorio y Auditoría de Seguridad (EDT 8.2)

Se ejecutó la auditoría automatizada `audit_repository_hygiene` (`src/closure/audit.py`) sobre todos los archivos del repositorio:
- **Credenciales y claves privadas:** **0 hallazgos** (el descargador `scripts/download_kaggle_datasets.py` consume variables de entorno en tiempo de ejecución sin almacenar secretos en código ni configuración).
- **Datos clínicos restringidos:** **0 hallazgos** (ningún archivo `.nii`, `.nii.gz` o `.dcm` clínico se encuentra versionado en Git; las reglas de `.gitignore` excluyen `data/raw/`, `data/processed/` y `local_*.csv`).
- **Archivos temporales y artefactos de compilación:** Excluidos mediante `.gitignore` (`.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`).

---

## 6. Consolidación de Indicadores de Seguimiento (S5 a S16)

| ID | Indicador | Meta Línea Base | Resultado Consolidado (S16) | Estado |
|---|---|---|---|---|
| **I1** | Actividades cumplidas | $\ge 85\text{ \%}$ | **100 %** (4/4 actividades de cierre y sustentación culminadas) | **Cumplido** |
| **I2** | Entregables aceptados | Según cronograma | **100,0 %** (7/7 formalmente aceptados: E1, E2, E3, E4, E5, E6 y E7) | **Cumplido** |
| **I3** | Hitos en fecha | $\ge 80\text{ \%}$ | **100 %** (7/7 hitos cumplidos en fecha: H1, H2, H3, H4, H5, H6 y H7) | **Cumplido** |
| **I4** | Desviación de esfuerzo | $-15\text{ \%}$ a $+20\text{ \%}$ | **0 % en S16** (24 h reales / 24 h planificadas; dentro de meta en todo el semestre) | **Cumplido** |
| **I5** | Riesgos altos abiertos | Tendencia descendente | **0 riesgos altos abiertos** (R1 a R8 completamente mitigados y cerrados) | **Cumplido** |
| **I6** | Incidencias resueltas | $\ge 80\text{ \%}$ al cierre | **100 %** de incidencias resueltas y verificadas con pruebas automatizadas | **Cumplido** |
| **I7** | Reproducibilidad | $100\text{ \%}$ final | **100 %** (**200/200 pruebas automatizadas aprobadas** y 9 runners de verificación) | **Cumplido** |
| **I8** | Rendimiento | Superar referencia | **No aplica en 2026-II** (alcance semestral excluye entrenamiento con datos reales) | **N/A (Justificado)** |
| **I9** | Portabilidad | Resultado documentado (S13) | **Documentado:** 327,30 MB RAM, ~585–601 ms CPU, **0,00 % degradación** en TorchScript | **Cumplido** |
| **I10** | Cobertura documental | $100\text{ \%}$ final | **100,0 %** (23/23 artefactos obligatorios entre configs, documentos y scripts verificados) | **Cumplido** |

---

## 7. Retrospectiva Técnica y Lecciones Aprendidas (EDT 8.4)

1. **Unificación temprana en un repositorio modular (CAM-001 en S5):** Consolidar todo el ciclo de vida en `alzheimer-vit-smri` bajo `uv.lock` evitó la fragmentación por entregables aislados y permitió construir una suite acumulativa de **200 pruebas unitarias e integración**.
2. **Adaptación ante restricciones de acceso clínico DUA (CAM-002 en S7 y ajuste S8):** La incorporación del adaptador para cortes axiales reales de Kaggle (OASIS y ADNI 4-Class) permitió validar el pipeline de preprocesamiento E2 frente a variaciones reales de nomenclatura y contraste sin infringir acuerdos de uso de datos hospitalarios.
3. **Diseño anticipado de interfaces de interpretabilidad (E3 $\to$ E5 en S8):** Exponer `get_attention_maps()` desde la concepción de `MultiHeadSelfAttention` permitió acoplar `AttentionRollout` en la Semana 12 y el prototipo `IntegratedSystemPipeline` en la Semana 13 sin refactorizar la arquitectura ViT-B/16.
4. **Rigor ético y metodológico (Mitigación de R6 y R7):** Codificar las advertencias `CLINICAL_DISCLAIMER` y `UNTRAINED_MODEL_NOTICE` directamente en las estructuras de datos de salida garantiza que ningún usuario o consumidor de software confunda una prueba funcional con un diagnóstico médico.

---

## 8. Protocolo de Demostración Técnica Repetible (EDT 8.5)

La demostración completa para la sustentación final se ejecuta con tres comandos deterministas desde la raíz del repositorio:

```bash
# 1. Ejecutar la auditoría integral de cierre formal E7 y sustentación (Semana 16)
uv run python scripts/run_s16_verification.py

# 2. Ejecutar la demostración interactiva de la interfaz técnica E6 con exportación de artefactos
uv run python scripts/run_e6_verification.py

# 3. Ejecutar la suite completa de 200 pruebas automatizadas
uv run pytest -q
```
