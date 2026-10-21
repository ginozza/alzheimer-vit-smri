# Entregable E4: Código Fuente del Sistema
## Documento Técnico de Aceptación y Consolidación (Hito H4 - Semana 11)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena - Programa de Ingeniería de Sistemas  
**Período:** Seminario III (2026-II)  
**Fecha de corte:** Semana 11 (12 a 18 de octubre de 2026)  
**Estado del entregable:** **Aceptado Formalmente (Hito H4 Cumplido al 100 %)**

---

## 1. Resumen Ejecutivo y Dictamen de Aceptación

El presente documento formaliza la entrega y aceptación del **Entregable E4: Código fuente del sistema**, correspondiente al paquete **EDT 5.0 (Componentes 5.1 a 5.5)** de la Línea Base del Proyecto.

### Evaluación del Criterio de Aceptación Oficial (Línea Base, Pág. 4):
> *«El código puede instalarse, importar los módulos e instanciar la arquitectura; las pruebas con tensores sintéticos verifican dimensiones y flujo sin entrenar el modelo.»*

| Criterio de Aceptación | Evidencia Implementada y Verificada | Dictamen |
|---|---|---|
| **Instalabilidad del código** | Paquete instalable `alzheimer-vit-smri==0.1.0` vía `pyproject.toml` y `uv pip install -e .`. Instalación limpia en 21 ms. | **SUPERADO** |
| **Importación de módulos** | Módulos raíz `src.data`, `src.model`, `src.training`, `src.inference` importables sin errores ni dependencias circulares. | **SUPERADO** |
| **Instanciación de arquitectura** | Modelo `VisionTransformer` (ViT-B/16 triclase) instanciado con 85.800.963 parámetros entrenables y forward pass verificado. | **SUPERADO** |
| **Verificación de dimensiones y flujo** | Tensor de entrada $(B, 3, 224, 224)$ `float32` $\rightarrow$ logits $(B, 3)$ finitos, auditado en 144 pruebas automatizadas. | **SUPERADO** |
| **Restricción académica (sin entrenar)** | Andamiaje de optimización y métricas probado con tensores sintéticos controlados sin generar pesos clínicos definitivos. | **SUPERADO** |

---

## 2. Arquitectura Modular del Repositorio

El código fuente se encuentra organizado bajo una arquitectura desacoplada en cuatro subpaquetes principales dentro de `src/`:

```
alzheimer-vit-smri/
├── pyproject.toml              # Definición del paquete alzheimer-vit-smri==0.1.0 y dependencias
├── configs/                    # Parámetros externos desacoplados (Criterio Transversal)
│   ├── e2.json                 # Normalización y clipping del preprocesamiento
│   ├── e3_architecture.json    # Hiperparámetros de la arquitectura ViT-B/16
│   ├── e4_training.json        # Optimizador AdamW, tasa de aprendizaje y clipping
│   ├── e4_dataset.json         # Lotes, subprocesos y validación tensorial
│   └── e4_inference.json       # Inferencia diagnóstica y umbrales de decisión
├── src/
│   ├── data/                   # EDT 2.0, 3.0 y 5.1: Carga, particionado, QC y DataLoader
│   │   ├── dataset.py          # MRI25DDataset y SyntheticMRIDataset
│   │   ├── dataloader.py       # create_mri_dataloader y mri_collate_fn
│   │   ├── inventory_loader.py # Ingesta y validación de metadatos clínicos
│   │   ├── kaggle_adapter.py   # Adaptador de neuroimágenes 2D reales
│   │   ├── preprocessing.py    # QC técnico y normalización de volúmenes
│   │   ├── slice_representation.py # Extractor de triplete canónico axial
│   │   └── subject_split.py    # Particionado estratificado sin fuga de sujetos
│   ├── model/                  # EDT 4.0 y 5.2: Arquitectura ViT-B/16 triclase
│   │   ├── config.py           # Dataclass ViTConfig con serialización JSON
│   │   ├── layers.py           # PatchEmbedding, MHSA, MLP, TransformerBlock, Head
│   │   └── vit.py              # Clase VisionTransformer y hooks de interpretabilidad
│   ├── training/               # EDT 5.3 y 5.4: Optimización, ciclo de entrenamiento y evaluación
│   │   ├── config.py           # TrainingConfig con validación de hiperparámetros
│   │   ├── engine.py           # train_batch con backward pass y gradient clipping
│   │   ├── evaluation.py       # EvaluationMetrics, matriz 3x3 y evaluate_model
│   │   ├── losses.py           # Entropía cruzada ponderada y balanceo por clase
│   │   ├── optimizers.py       # AdamW desacoplado con decaimiento de peso selectivo
│   │   └── trainer.py          # FutureTrainer, TrainingHistory y checkpoints .pt
│   └── inference/              # EDT 5.5: Pipeline de inferencia diagnóstica
│       └── predictor.py        # InferencePipeline y PredictionResult
├── scripts/                    # Runners de auditoría y verificación reproducible
│   ├── run_e1_verification.py  # Auditoría E1 (partición y no fuga)
│   ├── run_e2_demo.py          # Demostración E2 (preprocesamiento)
│   ├── run_e3_verification.py  # Auditoría E3 (arquitectura ViT-B/16)
│   ├── run_s9_verification.py  # Verificación estructural S9
│   ├── run_s10_verification.py # Verificación de integración S10
│   └── run_e4_verification.py  # Auditoría integral de aceptación E4 (Hito H4)
└── tests/                      # 144 pruebas unitarias automatizadas (100 % aprobadas)
```

---

## 3. Desglose Técnico de Componentes Implementados (EDT 5.0)

### 3.1 Carga de Datos y DataLoader (EDT 5.1)
- **`MRI25DDataset`:** Maneja tanto manifiestos CSV (`outputs.csv`) como DataFrames en memoria. Valida que cada corte axial `.npy` posea dimensiones estrictas $(3, 224, 224)$ `float32`, valores finitos (sin `NaN` ni `Inf`) y mapeo a clases triclase:
  $$\text{CN} \rightarrow 0, \quad \text{MCI} \rightarrow 1, \quad \text{AD} \rightarrow 2$$
- **`SyntheticMRIDataset`:** Genera cortes axiales sintéticos deterministas con firma de ruido por clase, permitiendo pruebas reproducibles sin I/O pesado.
- **`create_mri_dataloader` y `mri_collate_fn`:** Empaqueta muestras en tensores de lote $(B, 3, 224, 224)$ y $(B,)$, verificando tipos de datos y preservando identificadores de paciente (`Subject_ID`).

### 3.2 Modelo ViT-B/16 Triclase (EDT 5.2)
- Modelo basado en Dosovitskiy et al. (2020) con adaptación 2.5D tricanal sobre cortes axiales canónicos centrados en hipocampo.
- **Conteo exacto de parámetros entrenables:**
  - Patch Embedding ($16 \times 16$, $D=768$, 196 parches + `[CLS]` + pos embeddings): **742.656**
  - Codificador Transformer (12 bloques Pre-Norm, 12 cabezas de atención, MLP $3072$): **85.054.464**
  - Cabeza de Clasificación Triclase (LayerNorm + Linear a 3 clases): **3.843**
  - **TOTAL SISTEMA:** **85.800.963 parámetros entrenables**.

### 3.3 Entrenamiento Futuro y Gestión de Checkpoints (EDT 5.3)
- **`FutureTrainer`:** Orquesta pasadas de entrenamiento y validación por época mediante `fit(train_loader, val_loader, epochs)`.
- **Pérdida Balanceada:** `build_classification_loss` con pesos por frecuencia inversa para compensar el desbalance de clases de los datasets reales:
  $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
- **Optimizador AdamW:** Dos grupos de parámetros; aplica decaimiento de peso ($0.05$) a matrices convolucionales y lineales, excluyendo sesgos (*biases*), LayerNorms y embeddings posicionales.
- **Persistencia de Checkpoints:** Guarda y restaura de forma desacoplada `model_state_dict`, `optimizer_state_dict`, historial de entrenamiento y metadatos en archivos binarios `.pt`.

### 3.4 Módulo de Evaluación Diagnóstica Multiclase (EDT 5.4)
- **Matriz de Confusión $3 \times 3$:** Registra clasificaciones cruzadas entre `CN`, `MCI` y `AD`.
- **Métricas Multicriterio:** Exactitud global (*Accuracy*), Exactitud balanceada (*Balanced Accuracy*), Sensibilidad macro (*Macro Recall*), Precisión macro (*Macro Precision*), Especificidad macro (*Macro Specificity*) y F1-score macro (*Macro F1*).
- **Protección contra División por Cero:** Asigna $0.0$ automáticamente cuando el soporte o los positivos predichos son nulos, evitando excepciones en particiones dispersas.

### 3.5 Pipeline de Inferencia Diagnóstica (EDT 5.5)
- **`InferencePipeline`:** Ejecuta inferencia en modo `eval()` y con `torch.no_grad()`.
- **Modos de Entrada:** Inferencia unitaria sobre tensor (`predict_tensor`), por lote (`predict_batch`) y directa desde archivo en disco (`predict_file`).
- **Salida Estructurada (`PredictionResult`):** Clase predicha (`CN`, `MCI` o `AD`), índice entero, confianza numérica ($0.0$ a $1.0$), distribución completa Softmax y logits crudos.
- **Puente con Interpretabilidad (E5):** Captura opcional de tensores de atención en las 12 capas codificadoras ($(B, 12, 197, 197)$) para alimentar directamente el algoritmo de Attention Rollout en la Semana 12.

---

## 4. Evidencia de Verificación y Pruebas Automatizadas

### 4.1 Suite de Pruebas Unitarias e Interoperabilidad
El sistema cuenta con **144 pruebas unitarias automatizadas** que abarcan la totalidad de los entregables desarrollados hasta la fecha:

```
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: alzheimer-vit-smri
configfile: pyproject.toml
testpaths: tests
collected 144 items

tests/test_dataset.py ........                                           [  5%]
tests/test_evaluation.py ........                                        [ 11%]
tests/test_inference.py ......                                           [ 15%]
tests/test_interoperability.py ....                                      [ 18%]
tests/test_inventory.py ....                                             [ 20%]
tests/test_kaggle_adapter.py .....                                       [ 24%]
tests/test_preprocessing.py .................................            [ 47%]
tests/test_preprocessing_batch.py ..................                     [ 60%]
tests/test_slice_dimensions.py ...                                       [ 62%]
tests/test_subject_leakage.py ...                                        [ 64%]
tests/test_trainer.py .....                                              [ 68%]
tests/test_training.py .................                                  [ 79%]
tests/test_vit_architecture.py ...............................           [100%]

============================ 144 passed in 38.45s =============================
```

### 4.2 Runner de Auditoría Formal (`scripts/run_e4_verification.py`)
El script ejecuta una auditoría de 6 etapas que verifica formalmente el cumplimiento del criterio de aceptación de la Línea Base:
```bash
uv run python scripts/run_e4_verification.py
```
**Resultado de la auditoría:**
- Instalabilidad e importación: SUPERADA (4 subpaquetes operativos).
- Esquemas de configuración externa: SUPERADA (5 archivos JSON validados).
- Ingesta de datos y DataLoader: SUPERADA (lotes $4 \times 3 \times 224 \times 224$ float32).
- ViT-B/16 triclase: SUPERADA (85.800.963 parámetros y logits finitos).
- Entrenamiento y checkpoints: SUPERADA (ciclo multiextensión y guardado .pt).
- Evaluación e inferencia: SUPERADA (matriz $3 \times 3$, Softmax y 12 capas de atención).
- **Dictamen:** Criterio de aceptación cumplido al 100 %.

---

## 5. Trazabilidad con la Línea Base

| Resultado Esperado | Entregable | Criterio de Aceptación | Estado en S11 |
|---|---|---|---|
| **Implementar los modelos** | **E4** | *«El código puede instalarse, importar los módulos e instanciar la arquitectura; las pruebas con tensores sintéticos verifican dimensiones y flujo sin entrenar el modelo»* | **Aceptado Formalmente (Hito H4 cumplido)** |
| **Separación de datos** | E1 | *«División definida por sujeto con cero fuga»* | Cumplido y revalidado en `test_interoperability.py` |
| **Preprocesamiento uniforme** | E2 | *«Entradas de tamaño y formato uniformes»* | Cumplido: tensores $(3, 224, 224)$ `float32` |
| **Arquitectura especificada** | E3 | *«Adaptación 2.5D y salida triclase justificadas»* | Cumplido: ViT-B/16 con 85.8M parámetros |
| **Código fuente integrado** | E4 | *«Código instalado, módulos importados, flujo tensorial verificado»* | **Cumplido al 100 % (144 pruebas aprobadas)** |

---

## 6. Próximo Paso en el Cronograma (Semana 12)

Con el cierre y aceptación formal del **Entregable E4 (Hito H4)**, el proyecto habilita el inicio de:
> **Semana 12:** Implementación del módulo de Attention Rollout y documentación de su funcionamiento.  
> **Entregable a alcanzar:** **Entregable E5: Módulo de interpretabilidad**.  
> **Hito programado:** **Hito H5: E5 aceptado**.  
> **Responsable:** Juan Simancas.
