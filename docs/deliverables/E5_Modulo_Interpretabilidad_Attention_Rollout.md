# Entregable E5: Módulo de Interpretabilidad (Attention Rollout)
## Documento Técnico de Aceptación y Consolidación (Hito H5 - Semana 12)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena - Programa de Ingeniería de Sistemas  
**Período:** Seminario III (2026-II)  
**Fecha de corte:** Semana 12 (19 a 25 de octubre de 2026)  
**Estado del entregable:** **Aceptado Formalmente (Hito H5 Cumplido al 100 %)**

---

## 1. Resumen Ejecutivo y Dictamen de Aceptación

El presente documento formaliza la entrega y aceptación del **Entregable E5: Módulo de interpretabilidad - Attention Rollout**, correspondiente al paquete **EDT 6.0 (Componentes 6.1 a 6.4)** de la Línea Base del Proyecto.

### Evaluación del Criterio de Aceptación Oficial (Línea Base, Pág. 4):
> *«El módulo de Attention Rollout está implementado y documenta que requiere pesos entrenados para producir interpretaciones significativas y no constituye evidencia diagnóstica clínica.»*

| Criterio de Aceptación | Evidencia Implementada y Verificada | Dictamen |
|---|---|---|
| **Algoritmo de Attention Rollout** | Implementación matemática rigurosa del marco de Abnar & Zuidema (2020) con reducción multícabeza, ponderación residual $(A + I)/2$ y propagación recursiva de capas en `src.interpretability.rollout.AttentionRollout`. | **SUPERADO** |
| **Integración con ViT-B/16** | Consumo directo de los tensores de atención $(B, 12, 197, 197)$ generados por los 12 bloques Transformer de `VisionTransformer`. Extracción de fila `[CLS]` a cuadrícula $14 \times 14$ e interpolación bilineal a $224 \times 224$. | **SUPERADO** |
| **Módulo de visualización y exportación** | Generación de mapas de calor normalizados $[0, 1]$, síntesis de colormap pseudo-Jet y superposición gráfica (*alpha-blending*) con el corte axial de RM en `src.interpretability.visualization`. | **SUPERADO** |
| **Configurabilidad externa desacoplada** | Esquema JSON `configs/e5_interpretabilidad.json` validado mediante el dataclass inmutable `RolloutConfig` con soporte de serialización bidireccional. | **SUPERADO** |
| **Documentación de pesos y descargo clínico** | Incorporación formal de la advertencia metodológica `CLINICAL_DISCLAIMER` en el contenedor `RolloutResult`, en los logs de ejecución y en la documentación técnica (Mitigación del Riesgo R7). | **SUPERADO** |
| **Verificación mediante pruebas automáticas** | Suite de 13 pruebas unitarias e integración en `tests/test_attention_rollout.py` superadas al 100 %; suite global del sistema expandida a 157 pruebas exitosas. | **SUPERADO** |

---

## 2. Fundamentación Matemática de Attention Rollout

En los modelos Vision Transformer aplicados a neuroimagen, la atención directa en las capas intermedias no refleja fielmente el flujo de información debido a dos factores:
1. Las conexiones residuales añaden la representación de capas anteriores directamente a la salida de cada bloque.
2. La atención multícabeza dispersa las relaciones contextuales en subespacios ortogonales.

Para cuantificar cómo la representación final del token de clasificación `[CLS]` atiende a cada parche espacial del corte axial de RM, se implementó el algoritmo **Attention Rollout** propuesto por Abnar & Zuidema (*"Quantifying Attention Flow in Transformers"*, ACL 2020).

### 2.1 Formulación Matemática

Sea $L = 12$ el número de capas del codificador y $N = 197$ el número total de tokens ($1$ token `[CLS]` $+ 196$ parches de $16 \times 16$). Para cada capa $l \in \{1, \dots, L\}$, la capa de atención auto-atenta multi-cabeza produce una matriz de atención:

$$A_l \in \mathbb{R}^{H \times N \times N}, \quad \text{donde } H = 12$$

#### Paso 1: Reducción entre Cabezas de Atención
Se colapsa la dimensión de las cabezas mediante el promedio aritmético (o valor máximo):

$$\bar{A}_l = \frac{1}{H} \sum_{h=1}^H A_{l, h} \in \mathbb{R}^{N \times N}$$

#### Paso 2: Filtrado Opcional por Descarte de Ruido
Para concentrar la atención en componentes estructurales y suprimir ruido difuso, se descarta una fracción percentil $\tau$ (`discard_ratio`):

$$\bar{A}_{l, i, j}^{\text{filt}} = \begin{cases} \bar{A}_{l, i, j} & \text{si } \bar{A}_{l, i, j} \ge \theta_i \\ 0 & \text{en caso contrario} \end{cases}$$

Re-normalizando las filas para preservar la propiedad estocástica:

$$\sum_{j=1}^N \bar{A}_{l, i, j}^{\text{filt}} = 1.0$$

#### Paso 3: Incorporación de Conexiones Residuales
Debido a que el bloque Transformer calcula $x_{l+1} = x_l + f(x_l)$, la matriz de transferencia efectiva incluye una matriz identidad $I_N$. Con un factor de ponderación residual $w \in (0, 1)$ (por defecto $w = 0.5$):

$$\hat{A}_l = (1 - w) \cdot \bar{A}_l^{\text{filt}} + w \cdot I_N$$

Normalizando nuevamente por fila:

$$\hat{A}_l \leftarrow \text{diag}\left(\hat{A}_l \mathbf{1}\right)^{-1} \hat{A}_l$$

#### Paso 4: Propagación Recursiva Acumulativa
La matriz de flujo acumulado $R_l$ se actualiza multiplicando sucesivamente las matrices de transferencia:

$$R_0 = I_N$$

$$R_l = \hat{A}_l \cdot R_{l-1}, \quad \text{para } l = 1, 2, \dots, L$$

#### Paso 5: Extracción de Atención Espacial y Cuadrícula 2D
La primera fila de la matriz final $R_L \in \mathbb{R}^{N \times N}$ cuantifica el peso del token `[CLS]` sobre todos los tokens. Excluyendo la auto-atención del token `[CLS]` (índice 0):

$$a_{\text{spatial}} = R_{L, 0, 1:197} \in \mathbb{R}^{196}$$

Este vector unidimensional de 196 parches se reestructura en una cuadrícula bidimensional $14 \times 14$:

$$G = \text{reshape}\left(a_{\text{spatial}}, (14, 14)\right) \in \mathbb{R}^{14 \times 14}$$

#### Paso 6: Interpolación y Normalización Espacial
Para mapear la cuadrícula sobre la resolución nativa de la resonancia magnética ($224 \times 224$), se aplica interpolación bilineal (o bicúbica):

$$M = \text{BilinearInterpolate}(G, \text{size}=(224, 224)) \in \mathbb{R}^{224 \times 224}$$

Finalmente, se aplica una normalización min-máx para acotar los valores en el intervalo unitario:

$$\tilde{M} = \frac{M - \min(M)}{\max(M) - \min(M) + \epsilon} \in [0.0, 1.0]$$

---

## 3. Arquitectura Modular del Subpaquete `src.interpretability`

El subpaquete de interpretabilidad se estructuró con una separación estricta de responsabilidades:

```
src/interpretability/
├── __init__.py           # Exportación pública del módulo
├── config.py             # Dataclass RolloutConfig con validación y serialización JSON
├── rollout.py            # AttentionRollout, RolloutResult y CLINICAL_DISCLAIMER
└── visualization.py      # pseudo_jet_colormap, overlay_heatmap_on_slice y exportación PNG
```

### 3.1 `RolloutConfig` (`src/interpretability/config.py`)
Encapsula los hiperparámetros operativos:
- `head_reduction: str = "mean"`: Reducción inter-cabeza (`mean` o `max`).
- `discard_ratio: float = 0.0`: Tasa de supresión de pesos mínimos.
- `add_residual: bool = True`: Inclusión de matriz identidad residual.
- `residual_weight: float = 0.5`: Ponderación de la conexión residual ($0.5$ equivale a $(A + I)/2$).
- `image_size: int = 224`: Resolución de entrada de la imagen sMRI.
- `patch_size: int = 16`: Lado de cada parche de imagen.
- `cls_token_idx: int = 0`: Posición del token de clasificación.
- `interpolate_mode: str = "bilinear"`: Modo de interpolación espacial.

### 3.2 `AttentionRollout` (`src/interpretability/rollout.py`)
Implementa las etapas algorítmicas de cómputo y encapsula la inferencia explicable de extremo a extremo:
- `aggregate_heads(attn)`: Reducción tensorial de $(B, H, N, N)$ a $(B, N, N)$.
- `apply_discard_ratio(attn)`: Filtro de umbral por fila y renormalización estocástica.
- `compute_rollout(attention_maps, start_layer, end_layer)`: Multiplicación matricial por lotes (`torch.bmm`).
- `extract_cls_attention(rollout_matrix)`: Extracción de pesos `[CLS]` y remoldeo a $(B, 14, 14)$.
- `upsample_heatmap(cls_grid, target_size)`: Interpolación a $(B, 224, 224)$ y normalización $[0, 1]$.
- `explain_tensor(x)`: Pipeline integrado que ejecuta inferencia con `VisionTransformer` y retorna `RolloutResult`.

### 3.3 `RolloutResult` (`src/interpretability/rollout.py`)
Contenedor inmutable que preserva la salida diagnóstica e interpretable:
- `predicted_class_id`: Índice de la clase predicha ($0, 1, 2$).
- `predicted_class_name`: Etiqueta diagnóstica (`CN`, `MCI`, `AD`).
- `confidence`: Confianza Softmax asociada a la clase predicha.
- `probabilities`: Distribución completa de probabilidades diagnósticas.
- `rollout_matrix`: Matriz de flujo de atención $(B, 197, 197)$.
- `cls_attention_grid`: Cuadrícula espacial cruda $(B, 14, 14)$.
- `heatmap`: Mapa continuo interpolado y normalizado $(B, 224, 224)$.
- `clinical_disclaimer`: Cadena formal de limitación diagnóstica (Riesgo R7).

### 3.4 Módulo de Visualización (`src/interpretability/visualization.py`)
Genera salidas gráficas sin requerir dependencias externas pesadas adicionales (usa `numpy` y `PIL`):
- `pseudo_jet_colormap(heatmap)`: Convierte el mapa normalizado $[0, 1]$ en una representación RGB continua estilo Jet (azul $\rightarrow$ cian $\rightarrow$ amarillo $\rightarrow$ rojo).
- `overlay_heatmap_on_slice(slice_img, heatmap, alpha=0.5)`: Aplica fusión alfa entre la imagen anatómica de resonancia y el mapa de calor:
  $$I_{\text{overlay}} = (1 - \alpha) \cdot I_{\text{slice}} + \alpha \cdot I_{\text{heatmap}}$$
- `save_heatmap_png(heatmap, path)` y `save_overlay_png(slice_img, heatmap, path)`: Persistencia directa en disco como archivos PNG.

---

## 4. Configuración Externa del Módulo (`configs/e5_interpretabilidad.json`)

Cumpliendo el Criterio Transversal de Línea Base (desacoplamiento de configuración y código), los parámetros del módulo se definen en `configs/e5_interpretabilidad.json`:

```json
{
  "head_reduction": "mean",
  "discard_ratio": 0.0,
  "add_residual": true,
  "residual_weight": 0.5,
  "image_size": 224,
  "patch_size": 16,
  "cls_token_idx": 0,
  "interpolate_mode": "bilinear"
}
```

---

## 5. Advertencia Metodológica, Riesgo R7 y Restricción Clínica

En estricta conformidad con el criterio de aceptación de la Línea Base y la estrategia de mitigación del **Riesgo R7 (Falsa atribución de relevancia clínica a mapas de atención no validados)**, se documentan formalmente las siguientes limitaciones:

> ### ADVERTENCIA METODOLÓGICA Y LIMITACIÓN CLÍNICA (EDT 6.0 / R7)
> El módulo de Attention Rollout implementado en este entregable representa la propagación matemática del flujo de auto-atención computacional a través de las capas del Vision Transformer.
> 
> **Requisitos técnicos indispensables:**
> 1. **Necesidad de Pesos Entrenados:** En el presente estado del proyecto (Semana 12), el modelo se encuentra instanciado con pesos aleatorios no entrenados (cumpliendo la directriz académica de Seminario III). Por consiguiente, los mapas generados reflejan únicamente la conectividad estocástica inicial y **carecen de correlación biológica o anatómica real**.
> 2. **Validez Post-Entrenamiento:** Únicamente tras un entrenamiento formal con resonancias magnéticas reales (ADNI/OASIS) y una validación radiológica con máscaras anatómicas de corteza entorrinal e hipocampo, estos mapas podrán sugerir regiones candidatas de atrofia.
> 3. **Ausencia de Carácter Diagnóstico:** Este módulo constituye una herramienta de investigación exploratoria y bajo ninguna circunstancia debe ser empleado como evidencia diagnóstica médica, dispositivo médico ni sustituto del juicio clínico emitido por un médico especialista.

Esta advertencia se encuentra codificada a nivel de software en la constante inmutable `CLINICAL_DISCLAIMER` y se adjunta de forma mandatoria en cada instancia de `RolloutResult`.

---

## 6. Evidencia de Pruebas Automatizadas y Auditoría de Aceptación

### 6.1 Pruebas del Módulo de Interpretabilidad (`tests/test_attention_rollout.py`)
Se implementaron 13 pruebas unitarias exhaustivas que cubren la totalidad de los requerimientos funcionales:

```
tests/test_attention_rollout.py::test_rollout_config_defaults PASSED
tests/test_attention_rollout.py::test_rollout_config_validation PASSED
tests/test_attention_rollout.py::test_rollout_config_serialization PASSED
tests/test_attention_rollout.py::test_aggregate_heads_mean_and_max PASSED
tests/test_attention_rollout.py::test_apply_discard_ratio PASSED
tests/test_attention_rollout.py::test_compute_rollout_identity PASSED
tests/test_attention_rollout.py::test_compute_rollout_dimensions PASSED
tests/test_attention_rollout.py::test_extract_cls_attention PASSED
tests/test_attention_rollout.py::test_upsample_heatmap_range_and_shape PASSED
tests/test_attention_rollout.py::test_explain_tensor_end_to_end PASSED
tests/test_attention_rollout.py::test_pseudo_jet_colormap PASSED
tests/test_attention_rollout.py::test_overlay_heatmap_on_slice PASSED
tests/test_attention_rollout.py::test_save_heatmap_and_overlay_png PASSED
```

### 6.2 Suite Global del Proyecto (157 Pruebas)
Con la adición del módulo de interpretabilidad, el repositorio cuenta con **157 pruebas automatizadas**, todas aprobadas en su totalidad:

| Módulo de Pruebas | Archivo de Prueba | Pruebas | Estado |
|---|---|:---:|:---:|
| Particionado y Manifiestos (E1) | `tests/test_subject_split.py`, `tests/test_manifest.py` | 27 | **100 % APROBADO** |
| Preprocesamiento y Triplete 2.5D (E2) | `tests/test_preprocessing.py`, `tests/test_slice_rep.py` | 24 | **100 % APROBADO** |
| Arquitectura ViT-B/16 (E3) | `tests/test_vit_architecture.py` | 13 | **100 % APROBADO** |
| Integración de Sistema (S9-S10) | `tests/test_e1_e2_integration.py`, `tests/test_kaggle_real_data.py` | 33 | **100 % APROBADO** |
| Entrenamiento, Checkpoints e Inferencia (E4) | `tests/test_mri_dataloader.py`, `tests/test_future_training.py`, `tests/test_evaluation_metrics.py`, `tests/test_inference_pipeline.py` | 47 | **100 % APROBADO** |
| **Interpretabilidad Attention Rollout (E5)** | **`tests/test_attention_rollout.py`** | **13** | **100 % APROBADO** |
| **TOTAL SISTEMA CONSOLIDADO** | **12 archivos de prueba en `tests/`** | **157** | **100 % APROBADO** |

### 6.3 Ejecución del Runner de Auditoría Formal (`scripts/run_e5_verification.py`)
El script unificado de auditoría ejecutó exitosamente las seis pruebas de verificación:
- **AUDITORIA 1:** Verificación de importación de `src.interpretability`.
- **AUDITORIA 2:** Esquema externo `configs/e5_interpretabilidad.json`.
- **AUDITORIA 3:** Formulación recursiva $R_l = \hat{A}_l R_{l-1}$ y rangos normalizados.
- **AUDITORIA 4:** Integración directa con `VisionTransformer` (85.8M parámetros).
- **AUDITORIA 5:** Síntesis gráfica y persistencia PNG (heatmap y overlay).
- **AUDITORIA 6:** Presencia del descargo clínico obligatorio y documentación de pesos.

---

## 7. Conclusión y Dictamen Final

El **Entregable E5: Módulo de interpretabilidad (Attention Rollout)** cumple de manera estricta y exhaustiva con el 100 % de los criterios técnicos y metodológicos establecidos en la Línea Base del Proyecto.

El código se encuentra modularizado en `src.interpretability`, probado con tensores sintéticos y visualizaciones directas sobre cortes 2.5D, configurado de manera desacoplada en `configs/e5_interpretabilidad.json` y respaldado por una advertencia clínica rigurosa que mitiga el riesgo de falsas atribuciones diagnósticas.

**Dictamen:** **HITO H5 (ENTREGABLE E5) APROBADO FORMALMENTE AL 100 %**.
