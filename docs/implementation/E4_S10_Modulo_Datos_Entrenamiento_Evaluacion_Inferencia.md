# Implementación S10: Módulo de datos, entrenamiento futuro, evaluación e inferencia

## 1. Propósito y relación con la línea base

La Semana 10 (S10) corresponde a la actividad de la Línea Base:
*«Implementación del módulo de datos, configuración, entrenamiento futuro, evaluación e inferencia»*
con evidencia fijada como *«Código integrado de los principales componentes»* bajo la responsabilidad conjunta de ambos integrantes (Juan Simancas y Malak Sanchez).

Esta actividad aporta directamente al **Entregable E4: Código fuente del sistema** (paquete **EDT 5.0**). Al igual que en la Semana 9, E4 permanece formalmente abierto hasta la Semana 11 (Hito H4), donde se consolidarán las pruebas de interoperabilidad y la integración completa del sistema. Este documento formaliza el avance técnico y la interoperabilidad de los cinco componentes del paquete EDT 5.0 sin declarar el cierre anticipado del entregable.

El proyecto mantiene rigurosamente las restricciones metodológicas de la Línea Base: no se entrenan pesos definitivos, no se generan predicciones clínicas reales y no se evalúa rendimiento diagnóstico sobre pacientes. Toda la verificación se realiza mediante tensores sintéticos controlados y contratos tensoriales estandarizados.

---

## 2. Alcance técnico implementado (EDT 5.0)

### 2.1 Carga de datos y DataLoader (EDT 5.1)

Se implementaron dos clases de conjuntos de datos en `src/data/dataset.py` y utilidades de carga en `src/data/dataloader.py`:

- **`MRI25DDataset(Dataset)`:** Ingesta registros desde un manifiesto de preprocesamiento (DataFrame o archivo CSV generado en E2 como `outputs.csv`), validando:
  - Formato de tensor NumPy `.npy` con forma `(3, 224, 224)` y tipo `float32`.
  - Ausencia total de valores no finitos (`NaN` o `Inf`).
  - Mapeo estandarizado de etiquetas diagnósticas (`CN`: 0, `MCI`: 1, `AD`: 2).
  - Trazabilidad de metadatos clínicos (`Subject_ID`, `Session_ID`, `Cohort_Source`, `Diagnosis_Class`).
- **`SyntheticMRIDataset(Dataset)`:** Generador determinista de tensores MRI sintéticos para pruebas automatizadas y entornos CI/CD sin requerir lectura de disco.
- **`create_mri_dataloader(...)` y `mri_collate_fn`:** Factoría y función de agrupamiento (*collate*) que verifica que cada lote generado posea forma `(B, 3, H, W)` `float32` para imágenes y `(B,)` `int64` para etiquetas, preservando la lista de identificadores de sujeto.
- **Configuración externa:** `configs/e4_dataset.json` desacopla los parámetros de carga (tamaño de lote, subprocesos, tipo de dato y dimensiones).

### 2.2 Integración del modelo ViT-B/16 (EDT 5.2)

Se verificó el acoplamiento directo entre el flujo de datos del `DataLoader` y la arquitectura `VisionTransformer` (`src/model/vit.py`), confirmando:
- Ingesta directa de lotes `(B, 3, 224, 224)` sin transformaciones intermedias.
- Generación de logits `(B, 3)` para las tres clases objetivo (`CN`, `MCI`, `AD`).
- Conservación de los 85.800.963 parámetros entrenables definidos en E3.

### 2.3 Entrenamiento futuro y gestión de checkpoints (EDT 5.3)

El módulo `src/training/trainer.py` implementa el andamiaje del ciclo de entrenamiento:

- **`TrainingHistory`:** Estructura de datos que registra la evolución de pérdidas de entrenamiento (`train_loss`), pérdidas de validación (`val_loss`), exactitud (`val_accuracy`) y F1-score macro (`val_macro_f1`) a lo largo de las épocas, con serialización a formato JSON.
- **`FutureTrainer`:** Coordinador desacoplado que ejecuta:
  - `train_epoch(dataloader)`: Itera sobre los lotes utilizando `train_batch`, acumula la pérdida media y aplica el recorte de norma de gradiente configurado.
  - `validate_epoch(dataloader)`: Ejecuta una pasada de validación completa en modo de evaluación sin gradientes.
  - `fit(train_loader, val_loader, epochs)`: Orquesta el ciclo multiextensión guardando el historial.
  - `save_checkpoint(path, epoch, metrics)` y `load_checkpoint(path)`: Persiste y restaura el estado del modelo (`state_dict`), el optimizador (`state_dict`), la configuración y el historial.

### 2.4 Módulo de evaluación multiclase (EDT 5.4)

El módulo `src/training/evaluation.py` implementa métricas diagnósticas multiclase con protección estricta contra división por cero:

- **`compute_confusion_matrix`:** Matriz de confusión entera de $3 \times 3$ (filas: clase real, columnas: clase predicha).
- **`compute_per_class_metrics`:** Cálculo por clase de:
  - Precisión: $TP / (TP + FP)$
  - Sensibilidad / Exhaustividad (*Recall*): $TP / (TP + FN)$
  - Especificidad: $TN / (TN + FP)$
  - Valor F1: $2 \cdot P \cdot R / (P + R)$
  - Soporte (*Support*): número de muestras reales de la clase.
- **Métricas globales agregadas:**
  - Exactitud global (*Accuracy*): traza de la matriz sobre el total de muestras.
  - Exactitud balanceada (*Balanced Accuracy*): media macro de sensibilidades.
  - Promedios macro de precisión, sensibilidad, especificidad y F1.
- **`evaluate_model`:** Función integral que recorre un `DataLoader`, acumula predicciones y calcula la pérdida promedio y el objeto inmutable `EvaluationMetrics`.

### 2.5 Pipeline de inferencia (EDT 5.5)

El paquete `src/inference/` provee la abstracción de consumo para inferencia unitaria y por lote:

- **`PredictionResult`:** Contenedor estructurado que reporta:
  - `predicted_class_id`: índice entero (0, 1, 2).
  - `predicted_class_name`: etiqueta clínica legible (`CN`, `MCI`, `AD`).
  - `confidence`: probabilidad más alta del vector Softmax.
  - `probabilities`: diccionario con la distribución de probabilidades de las 3 clases (suma = 1.0).
  - `logits`: lista de puntuaciones crudas del modelo.
  - `attention_maps`: mapas de atención opcionales para integración con Attention Rollout (E5).
- **`InferencePipeline`:**
  - `predict_tensor(tensor)`: Inferencia sobre tensor individual 2.5D `(3, 224, 224)`.
  - `predict_batch(tensors)`: Inferencia eficiente sobre lotes `(B, 3, 224, 224)`.
  - `predict_file(path)`: Inferencia directa sobre archivos de corte axial `.npy` generados por el preprocesamiento E2.
  - Garantiza `model.eval()` y `torch.no_grad()`, rechazando entradas con dimensiones espaciales inválidas, rango incorrecto o valores `NaN`/`Inf`.
- **Configuración externa:** `configs/e4_inference.json`.

---

## 3. Evidencia verificable

La verificación de la Semana 10 se ejecuta de forma reproducible mediante el verificador integral y la suite de pruebas unitarias:

```bash
uv run python scripts/run_s10_verification.py
uv run pytest
```

### Resultados de la auditoría técnica S10 (`run_s10_verification.py`):
1. **Datos (5.1):** Lote sintético `(4, 3, 224, 224)` `float32` y etiquetas `(4,)` `int64` generado y validado.
2. **Modelo (5.2):** ViT-B/16 acoplado directamente al lote produciendo logits `(2, 3)` finitos.
3. **Entrenamiento futuro (5.3):** Ciclo de 2 épocas ejecutado sin errores, con serialización y restauración completa de checkpoint.
4. **Evaluación (5.4):** Matriz de confusión $3 \times 3$, cálculo de métricas macro y por clase verificado.
5. **Inferencia (5.5):** Inferencia unitaria y por lote ejecutada, con extracción de probabilidades Softmax y captura de mapas de atención en 12 capas.

### Estado de la suite de pruebas:
- **140 pruebas unitarias aprobadas al 100 %** (113 previas + 27 nuevas pruebas de S10).
- Cobertura completa de datasets, cargadores, cálculo matricial de métricas, entrenamiento futuro e inferencia.

---

## 4. Criterios de aceptación y estado de la Línea Base

- **Evidencia S10 cumplida:** *«Código integrado de los principales componentes»* implementado y probado en el repositorio.
- **Entregable E4:** Permanece formalmente en estado de avance técnico (Semana 10 de 11). Su aceptación e hito formal (**H4**) están programados para la Semana 11, tras la integración final de pruebas de interoperabilidad.
- **Restricción académica:** Se confirma que ningún componente produce pesos definitivos de entrenamiento ni genera afirmaciones clínicas diagnósticas.
