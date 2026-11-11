# Entregable E6: Prototipo de Integración y Manual Técnico de Uso
## Documento Técnico de Aceptación y Manual de Usuario (Hito H6 - Semana 14)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena - Programa de Ingeniería de Sistemas  
**Período:** Seminario III (2026-II)  
**Fecha de corte:** Semana 14 (02 a 08 de noviembre de 2026)  
**Estado del entregable:** **Aceptado Formalmente (Hito H6 Cumplido al 100 %)**

---

## 1. Resumen Ejecutivo y Dictamen de Aceptación

El presente documento formaliza la entrega y aceptación del **Entregable E6: Prototipo de integración**, correspondiente al paquete **EDT 7.0 (Componentes 7.1 a 7.5)** de la Línea Base del Proyecto, e incluye el **Manual Técnico y de Uso** exigido por los criterios transversales de documentación (Sección 6.1 y Objetivo Específico 7 de la Línea Base).

### Evaluación del Criterio de Aceptación Oficial (Línea Base, Sección 6, Pág. 4):
> *«La interfaz recibe una entrada de prueba, ejecuta el flujo técnico simulado, controla errores e indica que no contiene un modelo entrenado ni genera diagnósticos.»*

| Criterio de Aceptación Oficial | Evidencia Implementada y Verificada | Dictamen |
|---|---|---|
| **Recepción de entrada de prueba** | `TechnicalIntegrationInterface` y su CLI reciben muestras sintéticas deterministas (`CN`, `MCI`, `AD`), archivos `.npy` de cortes 2.5D y tensores en memoria $(3, 224, 224)$ `float32`. | **SUPERADO** |
| **Ejecución del flujo técnico simulado** | Tubería extremo a extremo `IntegratedSystemPipeline` que conecta validación 2.5D (E1/E2) $\rightarrow$ inferencia triclase ViT-B/16 (E3/E4) $\rightarrow$ Attention Rollout y superposición visual (E5). | **SUPERADO** |
| **Control estructurado de errores** | Validación exhaustiva y captura controlada (`IntegrationInputError`) para 9 categorías de fallo sin excepciones no controladas, retornando `status="ERROR"`, `error_code` y código de salida CLI `1`. | **SUPERADO** |
| **Aviso explícito de modelo no entrenado y ausencia de diagnóstico** | Banderas inmutables `is_trained_model=False` y `generates_clinical_diagnosis=False`, junto con `UNTRAINED_MODEL_NOTICE` y `CLINICAL_DISCLAIMER` en el objeto de salida, JSON y reporte en consola. | **SUPERADO** |
| **Manual de uso y demostración repetible** | Manual técnico de instalación, configuración, uso, control de errores y entrenamiento futuro integrado en este documento, acompañado del verificador `scripts/run_e6_verification.py`. | **SUPERADO** |

---

## 2. Arquitectura del Prototipo de Integración (EDT 7.0)

El subpaquete `src/integration/` consolida la capa de integración y exposición técnica del sistema:

```
src/integration/
├── __init__.py       # API pública del módulo de integración (EDT 7.0)
├── config.py         # Dataclass inmutable IntegrationConfig (configs/e6_integration.json)
├── pipeline.py       # IntegratedSystemPipeline, generate_synthetic_25d_case e IntegrationExecutionResult
├── interface.py      # TechnicalIntegrationInterface y CLI ejecutable (build_cli_parser, run_cli)
└── portability.py    # Benchmark formal del Indicador I9 (Tamaño, latencia y degradación)
```

---

## 3. Manual de Instalación, Configuración y Uso (EDT 7.4)

### 3.1 Requisitos del Sistema e Instalación Reproducible

El proyecto utiliza [uv](https://docs.astral.sh/uv/) con archivo de bloqueo determinista (`uv.lock`) sobre **Python 3.12** y **PyTorch $\ge$ 2.0.0**.

```bash
# 1. Clonar o ubicarse en la raíz del repositorio
cd alzheimer-vit-smri

# 2. Sincronizar el entorno virtual bloqueado (incluyendo dependencias de desarrollo)
uv sync --locked

# 3. Verificar la suite completa de pruebas unitarias e integración
uv run --locked pytest
```

### 3.2 Archivos de Configuración Externa (`configs/`)

Todos los hiperparámetros del sistema se encuentran desacoplados del código fuente:

| Archivo | Propósito |
|---|---|
| `configs/e2.json` | Geometría MNI152 ($z = -12\text{ mm}$), espaciado voxel ($1,0\text{ mm}$), clipping $p_1/p_{99}$ y normalización z-score. |
| `configs/e3_architecture.json` | Hiperparámetros de `VisionTransformer` (ViT-B/16, $224 \times 224$, parche $16 \times 16$, 12 capas, 12 cabezas, 3 clases). |
| `configs/e4_dataset.json` | Tamaño de lote, subprocesos y esquema tensorial $(3, 224, 224)$ `float32`. |
| `configs/e4_training.json` | Optimizador AdamW ($\text{lr}=3 \times 10^{-4}$, $\text{weight\_decay}=0.05$, `gradient_clip_norm=1.0`). |
| `configs/e4_inference.json` | Dispositivo de inferencia (`cpu`), clases objetivo (`CN`, `MCI`, `AD`) y extracción de atención. |
| `configs/e5_interpretabilidad.json` | Parámetros de Attention Rollout (`head_reduction="mean"`, `residual_weight=0.5`, interpolación `bilinear`). |
| `configs/e6_integration.json` | Parámetros de la interfaz de integración E6, semilla determinista (`42`), banderas de modelo no entrenado y umbrales de portabilidad I9. |

### 3.3 Ejecución del Prototipo de Integración desde Línea de Comandos (CLI)

#### Caso A: Ejecución con muestra sintética generada automáticamente (`CN`, `MCI` o `AD`)
```bash
uv run --locked python -c "import sys; from src.integration import run_cli; sys.exit(run_cli()[0])" \
  --synthetic-class MCI \
  --seed 42 \
  --subject-id DEMO_MCI_001 \
  --output-dir data/processed/demo_e6_mci
```

#### Caso B: Ejecución sobre un archivo tensor `.npy` externo $(3, 224, 224)$ `float32`
```bash
uv run --locked python -c "import sys; from src.integration import run_cli; sys.exit(run_cli()[0])" \
  --input-file data/processed/demo_e6_mci/DEMO_MCI_001_input_25d.npy \
  --subject-id PACIENTE_SIM_001 \
  --output-dir data/processed/demo_e6_external
```

#### Caso C: Uso programático desde Python
```python
from src.integration import TechnicalIntegrationInterface

interface = TechnicalIntegrationInterface(config_path="configs/e6_integration.json")
result = interface.execute_test_input(
    synthetic_class="AD",
    seed=42,
    subject_id="SUBJ_DEMO_AD",
    output_dir="data/processed/demo_programatico",
)
print(interface.format_execution_report(result))
```

### 3.4 Artefactos Generados por Ejecución

Cuando se especifica `--output-dir`, la interfaz exporta cuatro artefactos verificables:
1. `<subject_id>_input_25d.npy`: Tensor 2.5D normalizado $(3, 224, 224)$ `float32`.
2. `<subject_id>_attention_heatmap.png`: Mapa de calor $224 \times 224$ con colormap pseudo-Jet.
3. `<subject_id>_attention_overlay.png`: Superposición $\alpha$-blending ($\alpha = 0.5$) sobre el corte axial central.
4. `<subject_id>_integration_result.json`: Reporte estructurado con probabilidades, logits, latencia y descargos obligatorios.

---

## 4. Catálogo de Control de Errores de la Interfaz (EDT 7.2)

La interfaz técnica intercepta cualquier anomalía de entrada y devuelve un objeto `IntegrationExecutionResult` con `status="ERROR"` y código de salida `1` en CLI, evitando caídas abruptas:

| Código de Error (`error_code`) | Condición que lo Dispara | Comportamiento de la Interfaz |
|---|---|---|
| `FILE_NOT_FOUND` | La ruta `--input-file` no existe o es un directorio. | Reporta ruta inexistente y conserva avisos E6. |
| `UNSUPPORTED_FILE_EXTENSION` | Archivo con extensión distinta de `.npy` (p. ej. `.txt`, `.csv`). | Rechaza formato no soportado antes de lectura binaria. |
| `CORRUPT_NPY_FILE` | Archivo `.npy` truncado o con cabecera binaria inválida. | Captura excepción de E/S y reporta archivo corrupto. |
| `INVALID_INPUT_TYPE` | Objeto en memoria que no es `np.ndarray` ni `torch.Tensor`. | Rechaza tipo de objeto incompatible. |
| `EMPTY_INPUT_TENSOR` | Tensor con 0 elementos (`size == 0`). | Rechaza tensor vacío. |
| `NON_FINITE_VALUES` | Presencia de valores `NaN`, `+Inf` o `-Inf`. | Rechaza tensor por violación numérica E2. |
| `INVALID_DIMENSIONS` | Tensor con rango distinto de 3D o 4D (p. ej. matriz 2D). | Exige representación tricanal 2.5D. |
| `UNSUPPORTED_BATCH_SIZE` | Tensor 4D con tamaño de lote $B \ne 1$ en interfaz unitaria. | Informa que la interfaz unitaria opera sobre $B = 1$. |
| `SHAPE_MISMATCH` | Dimensiones distintas de $(3, 224, 224)$ (p. ej. $(3, 128, 128)$). | Rechaza geometría incompatible con `PatchEmbedding`. |
| `DEGENERATE_CONSTANT_SIGNAL` | Corte con desviación estándar nula ($\sigma < 10^{-8}$, imagen vacía/constante). | Rechaza muestra que no supera control de calidad. |
| `INVALID_TARGET_CLASS` | Clase sintética fuera de `("CN", "MCI", "AD")`. | Informa las tres clases válidas del sistema. |

---

## 5. Procedimiento Documentado para Entrenamiento Futuro (Objetivo Específico 7)

Conforme al alcance académico de **Seminario III (2026-II)**, durante el presente semestre **no se entrena el modelo ni se generan pesos clínicos**. Cuando se disponga de infraestructura GPU y cohortes autorizadas (ADNI / OASIS), el procedimiento reproducible de entrenamiento es el siguiente:

1. **Particionado E1 sin fuga de sujetos (`SubjectSplitter`):**
   - Dividir el inventario maestro a nivel de `Subject_ID` garantizando $S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$.
2. **Preprocesamiento E2 (`scripts/run_e2_preprocessing.py` o `scripts/run_e2_kaggle_pipeline.py`):**
   - Generar los tensores `.npy` $(3, 224, 224)$ `float32` y el manifiesto `outputs.csv` con control de calidad documentado.
3. **Cálculo de pesos de clase exclusivamente sobre `train` (`balanced_class_weights`):**
   - Calcular $w_c = N_{\text{total}} / (3 \cdot N_c)$ usando únicamente el conteo de etiquetas de la partición de entrenamiento y actualizar `class_weights` en `configs/e4_training.json`.
4. **Ciclo de entrenamiento (`FutureTrainer` en `src/training/trainer.py`):**
   - Instanciar `MRI25DDataset` y `create_mri_dataloader` para `Train` y `Validation`.
   - Instanciar `VisionTransformer(ViTConfig.load("configs/e3_architecture.json"))`.
   - Configurar `build_classification_loss` y `build_optimizer` (AdamW con exclusión de weight decay en biases, LayerNorm, token `[CLS]` y embeddings posicionales).
   - Ejecutar `trainer.fit(train_loader, val_loader, epochs=N)` y persistir checkpoints mediante `trainer.save_checkpoint(...)`.
5. **Evaluación e Interpretabilidad Post-Entrenamiento (`evaluate_model` y `AttentionRollout`):**
   - Evaluar la partición `Test` reservada con `evaluate_model` (`EvaluationMetrics`) y auditar los mapas de atención con `AttentionRollout` frente a regiones anatómicas del lóbulo temporal medial e hipocampo.

---

## 6. Advertencia Metodológica y Ausencia de Diagnóstico Clínico

> **AVISO OBLIGATORIO DE PROTOTIPO TÉCNICO (ENTREGABLE E6 / LÍNEA BASE):**  
> Esta interfaz ejecuta un flujo técnico simulado con datos de prueba y **NO contiene un modelo entrenado ni genera diagnósticos clínicos**. Los pesos del modelo ViT-B/16 corresponden a inicialización no entrenada conforme al alcance académico del periodo 2026-II; las probabilidades y mapas generados verifican exclusivamente la integridad dimensional, funcional y computacional del sistema.

---

## 7. Demostración Técnica Reproducible (EDT 7.5)

Para ejecutar la auditoría completa y demostración reproducible del Entregable E6 (Hito H6):

```bash
uv run --locked python scripts/run_s13_verification.py
uv run --locked python scripts/run_e6_verification.py
```

**Dictamen:** **HITO H6 (ENTREGABLE E6: PROTOTIPO DE INTEGRACIÓN) APROBADO FORMALMENTE AL 100 %**.
