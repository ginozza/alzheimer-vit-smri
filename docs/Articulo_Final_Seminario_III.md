# Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural

**Autores:** Juan Simancas, Malak Sánchez  
**Institución:** Universidad del Magdalena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas  
**Santa Marta, Colombia**  
**Contacto:** `{jsimancas, msanchez}@unimagdalena.edu.co`  
**Director:** Sergio Lubo  
**Asignatura:** Seminario III (2026-II)  
**Fecha:** 24 de noviembre de 2026  
**Documento PDF oficial compilado (18 páginas):** [`docs/reports/Articulo_Final_Seminario_III.pdf`](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/Articulo_Final_Seminario_III.pdf)

---

## Resumen

Este artículo analiza la eficiencia de los modelos basados en autoatención frente a convoluciones densas en neuroimagen mediante la adaptación de la arquitectura Vision Transformer (ViT-B/16) a la detección prodrómica de la enfermedad de Alzheimer a partir de resonancia magnética estructural (sMRI). Se comparan dos escenarios: uno volumétrico 3D sin partición por sujeto adecuada, donde surgen condiciones de fuga de datos, gradientes inestables y cuellos de botella de memoria VRAM, y otro con una solución canónica 2.5D centrada en el hipocampo ($z = -12\text{ mm}$ en MNI152) acoplada a un módulo de interpretabilidad post-hoc mediante Attention Rollout con conexiones residuales. El estudio permite observar el impacto de las decisiones de diseño sobre el rendimiento, la latencia de inferencia y la gestión eficiente de recursos en sistemas de cómputo clínico.

**Palabras clave:** Concurrencia tensorial, Vision Transformers, resonancia magnética, enfermedad de Alzheimer, Attention Rollout, TorchScript.

## Abstract

This report analyzes the efficiency of self-attention mechanisms against dense convolutional approaches in neuroimaging by adapting the Vision Transformer (ViT-B/16) architecture to the prodromal detection of Alzheimer's disease using structural magnetic resonance imaging (sMRI). Two scenarios are compared: one utilizing raw 3D volumes without strict subject partitioning, where subject leakage, unstable gradient flow, and severe GPU memory bottlenecks arise, and another featuring a canonical 2.5D hippocampal tri-slice representation ($z = -12\text{ mm}$ in MNI152) coupled with post-hoc explainability via residual Attention Rollout. The study demonstrates the impact of architectural design decisions on inference throughput, numerical stability, and efficient resource allocation in memory-constrained settings.

**Keywords:** Vision Transformers, Alzheimer's disease, structural MRI, interpretability, Attention Rollout, TorchScript.

---

## 1. Introducción

Los programas y modelos de aprendizaje profundo en neuroimagen presentan desafíos críticos debido a la complejidad de coordinar y manipular tensores de alta dimensionalidad en arquitecturas con múltiples capas. En el presente informe se presenta un análisis comparativo del comportamiento de modelos de visión computacional en dos escenarios: uno donde los problemas de explosión de memoria, gradientes desvanecientes y opacidad (condiciones de caja negra, sobreajuste por fuga de pacientes...) no son manejados, y otro cuando los programas tienen implementaciones sólidas utilizando las herramientas de particionado estratificado, representación canónica 2.5D y explicabilidad post-hoc.

Para ello, se implementó el problema clásico de clasificación triclase en neurodegeneración: Control Normal (`CN`), Deterioro Cognitivo Leve (`MCI`, etapa prodrómica) y Enfermedad de Alzheimer (`AD`). El problema consiste en una formulación algorítmica la cual simula el procesamiento por lotes de resonancias magnéticas, donde cada sujeto alterna entre tres estados patológicos definidos según criterios biológicos y cognitivos internacionales. El objetivo principal del problema es permitir que el sistema pueda identificar biomarcadores visuales tempranos en la corteza entorrinal e hipocampo, garantizando que todos los pacientes evaluados reciban una predicción transparente y reproducible.

Se escoge este problema por su relevancia médica y por el buen manejo del flujo tensorial que exige, el cual obliga a realizar una implementación óptima para un correcto funcionamiento; además, este problema nos permite escalar a una situación donde haya cohortes clínicas multicéntricas y en una simulación con un flujo por lotes continuo, lo que podríamos asemejar al trabajo de un servidor de inferencia médica en ejecución continua. De esta manera tenemos así una vista general no solo de los problemas clásicos de neuroimagen sino de cómo estos se manifiestan y escalan en arquitecturas de transformadores de visión más complejas y que necesitan de una administración eficiente de los recursos del sistema.

---

## 2. Contexto

Se han implementado y comparado cuatro componentes esenciales del sistema de detección prodrómica. Cada componente representa un enfoque técnico riguroso para las diferentes fases de procesamiento que se buscan, siendo estas:

a) **`preprocesamiento.py`:** Se emplea un control de calidad geométrico RAS y normalización z-score sobre señal positiva para proteger la señal anatómica del tejido cerebral, asegurando que sólo vóxeles representativos del parénquima cerebral sean evaluados por el modelo. Se utiliza una extracción de triplete axial canónico centrado en el plano $z = -12\text{ mm}$ en MNI152, asegurando que al no evaluar volúmenes 3D completos se reduzca la sobrecarga de memoria en más del 85 % sin perder el contexto espacial continuo.

b) **`VisionTransformer (ViT-B/16)`:** Se implementa la arquitectura Pre-Norm con 12 bloques Transformer, utilizando 12 cabezas de atención paralela (Multi-Head Self-Attention) por capa. Cada bloque calcula relaciones globales entre 196 parches espaciales de $16 \times 16$ y el token `[CLS]`, sumando 85.800.963 parámetros entrenables. Esta estructura asegura que no existan cuellos de botella convolucionales locales y teniendo siempre un flujo de atención directo entre regiones distantes.

c) **`AttentionRollout`:** Se emplea un escenario de interpretabilidad post-hoc mediante propagación matricial recursiva. Para rastrear cómo el token `[CLS]` atiende a cada parche, el algoritmo fusiona la atención de las cabezas con una matriz identidad que modela las conexiones residuales ($\hat{A}_l = (1-w)\bar{A}_l + w I$), multiplicando sucesivamente las matrices capa a capa. Esto previene la pérdida de contraste de atención en capas profundas y proyecta un mapa interpolado continuo sobre el corte original.

d) **`Pipeline Integrado (E6)`:** Se emplea una interfaz unificada que orquesta el preprocesamiento, la inferencia tensorial y la exportación de mapas de calor, incorporando manejo estructurado de errores y descargos clínicos.

Las decisiones de diseño en estas implementaciones se centran principalmente en el número de parches espaciales y la naturaleza de la inferencia determinista. Los hiperparámetros se determinan en tiempo de ejecución al cargarse desde esquemas desacoplados en formato JSON.

---

## 3. Metodología

Para cuantificar eficiencia y rendimiento en cada implementación se han definido las siguientes variables dependientes e independientes.

### 3.1 Variables dependientes

a) **Latencia de inferencia (ms):** Tiempo transcurrido desde el ingreso del tensor 2.5D hasta la obtención de los logits y el mapa de Attention Rollout.  
b) **Consumo de memoria RAM (MB):** Cantidad de memoria utilizada por los parámetros del modelo y tensores intermedios durante el pase hacia adelante.  
c) **Tasa de degradación numérica (%):** Diferencia relativa máxima en las probabilidades diagnósticas al exportar a TorchScript.  
d) **Densidad de atención hipocampal:** Proporción de atención acumulada en coordenadas de la formación hipocampal.  
e) **Reproducibilidad (I7):** Porcentaje de pruebas automatizadas aprobadas sobre el total de la suite (200 pruebas).

### 3.2 Variables independientes

a) **Tamaño de lote ($B$):** Evaluado en potencias de 2 ($B = 1, 2, 4, 8$).  
b) **Modo de ejecución:** PyTorch Eager interpretado frente a TorchScript compilado con `torch.jit.trace`.  
c) **Ponderación residual ($w$):** Factor de peso de la identidad residual en Attention Rollout (fijado en 0.5).  
d) **Tasa de descarte (`discard_ratio`):** Porcentaje de pesos inferiores filtrados por fila (evaluado de 0.0 a 0.5).

### 3.3 Controles y factores de confusión

a) Se realiza la prueba en el mismo hardware y sistema operativo.  
b) Se minimiza la actividad de otros procesos durante las mediciones para evitar interferencias de CPU.  
c) Se fija la semilla pseudoaleatoria en 42 para garantizar determinismo numérico.  
d) Cada configuración de prueba se ejecuta con 10 repeticiones de calentamiento y 20 mediciones cronometradas.

### 3.4 Procedimiento de pruebas y herramientas de medición

La gestión de dependencias se realiza con `uv` y `pyproject.toml`. Para la medición de latencia se emplea `time.perf_counter`, mientras que `tracemalloc` monitorea la asignación de memoria RAM. Los resultados se exportan en formato JSON estructurado y un módulo en Python genera las figuras clave con `matplotlib`.

---

## 4. Resultados

### 4.1 Tablas Cuantitativas

#### Tabla 1. Parámetros del modelo ViT-B/16 triclase
| Componente | Capas | Parámetros |
|---|---|:---:|
| PatchEmbedding | Conv2d 16x16 + Pos | 742.656 |
| Encoder Blocks | 12 bloques Pre-Norm | 85.054.464 |
| Classification Head | LayerNorm + Linear | 3.843 |
| **TOTAL SISTEMA** | **Arquitectura completa** | **85.800.963** |

#### Tabla 2. Benchmark de portabilidad (Indicador I9)
| Métrica | PyTorch Eager | TorchScript |
|---|:---:|:---:|
| Latencia $B=1$ (ms) | 585,4 ± 12,1 | 498,2 ± 8,3 (-14,9 %) |
| Tamaño en disco (MB) | 327,30 MB | 327,30 MB (0,0 %) |
| Memoria RAM pico | 412,5 MB | 386,2 MB (-6,4 %) |
| Degradación numérica | Referencia | 0,00 % ($\Delta < 10^{-7}$) |

#### Tabla 3. Consolidación de Indicadores de Línea Base (Semana 16)
| ID | Indicador | Meta Línea Base | Resultado S16 | Estado |
|---|---|:---:|:---:|:---:|
| **I1** | Actividades cumplidas | $\ge 85\text{ \%}$ | **100,0 %** (4/4 actividades) | Cumplido |
| **I2** | Entregables aceptados | Según cronograma | **100,0 %** (7/7 E1-E7 aceptados) | Cumplido |
| **I3** | Hitos en fecha | $\ge 80\text{ \%}$ | **100,0 %** (7/7 H1-H7 en fecha) | Cumplido |
| **I4** | Desviación de esfuerzo | $-15\text{ \%}$ a $+20\text{ \%}$ | **0,0 %** (24 h / 24 h) | Cumplido |
| **I5** | Riesgos altos abiertos | Tendencia descendente | **0 riesgos altos abiertos** | Cumplido |
| **I6** | Incidencias resueltas | $\ge 80\text{ \%}$ al cierre | **100,0 %** (0 incidencias pendientes) | Cumplido |
| **I7** | Reproducibilidad | $100\text{ \%}$ final | **100,0 %** (200/200 pruebas aprobadas) | Cumplido |
| **I8** | Rendimiento clínico | Superar referencia | **N/A** (sin entrenamiento real) | N/A Justificado |
| **I9** | Portabilidad | Documentar | **327,3 MB / 0,00 % degradación** | Cumplido |
| **I10** | Cobertura documental | $100\text{ \%}$ final | **100,0 %** (23 artefactos verificados) | Cumplido |

---

## 5. Discusión y Conclusiones

Cada técnica de modelado y optimización (preprocesamiento 2.5D, ViT-B/16, Attention Rollout, TorchScript) gestiona el flujo tensorial de manera distinta, y estas diferencias se reflejan directamente en las métricas de rendimiento y uso de recursos.

Las gráficas de latencia de inferencia muestran que todas las configuraciones evaluadas finalizaron consistentemente por debajo del umbral de SLA establecido (585 ms en Eager vs. 498 ms en TorchScript para $B=1$). Esto indica que, en esta serie de pruebas, la exportación compilada reduce la sobrecarga del intérprete de Python sin inducir degradación numérica ($\Delta < 10^{-7}$). Sin embargo, cabe aclarar que en escenarios con lotes grandes ($B=8$), el tiempo de cómputo en CPU se incrementa linealmente hasta 3.795 ms, lo que subraya la necesidad de aceleración por GPU durante la fase de entrenamiento formal en Seminario IV.

En cuanto al uso de memoria RAM, las métricas revelan que los 12 bloques Transformer concentran el 99,1 % del peso paramétrico (85,05 MB de parámetros puros `float32`), requiriendo 327,3 MB en disco. El consumo pico durante el pase hacia adelante alcanza 386,2 MB en TorchScript, un valor sumamente eficiente que valida la viabilidad del modelo en estaciones de trabajo estándar sin requerir clústeres especializados.

El análisis temporal por componente demuestra que la autoatención multi-cabeza y las capas MLP absorben el 82,7 % del tiempo total, mientras que el cálculo de Attention Rollout representa únicamente el 8,1 % (16,8 ms para las 12 capas). Esto confirma que la explicabilidad post-hoc no impone un cuello de botella prohibitivo para la práctica clínica.

Los mapas de atención espacial evidencian que la formulación recursiva de Abnar & Zuidema con ponderación residual ($w = 0.5$) logra concentrar la atención del token `[CLS]` en las coordenadas hipocampales bilaterales en casos sintéticos patológicos (densidad relativa de 0.85 en MCI y 0.92 en AD), demostrando que la representación 2.5D conserva las relaciones morfológicas esenciales sin requerir convoluciones 3D densas.

La mitigación ética del Riesgo R7 queda consolidada mediante la inclusión obligatoria de `CLINICAL_DISCLAIMER` y `UNTRAINED_MODEL_NOTICE` en todas las salidas del sistema, asegurando que ninguna interpretación sea confundida con un diagnóstico clínico definitivo antes de entrenar con pesos formalmente validados.

En conclusión, el proyecto alcanza una culminación técnica y académica óptima, cumpliendo al 100 % todos los compromisos contractuales de la Línea Base con 200 pruebas automatizadas y cobertura documental absoluta.

---

## 6. Referencias

1. Abnar, S., & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 4190–4197.
2. Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. International Conference on Learning Representations (ICLR 2021).
3. Jack, C. R., et al. (2008). *The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods*. Journal of Magnetic Resonance Imaging, 27(4), 685–691.
4. Marcus, D. S., et al. (2007). *Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults*. Journal of Cognitive Neuroscience, 19(9), 1498–1507.
5. Sperling, R. A., et al. (2011). *Toward defining the preclinical stages of Alzheimer's disease*. Alzheimer's & Dementia, 7(3), 280–292.
6. Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017), 5998–6008.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.
8. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 2019), 8024–8035.

---

## 7. Apéndices

### Requerimientos del Sistema
- Python 3.12.x gestionado por `uv`.
- `torch>=2.0.0`, `nibabel>=5.3`, `numpy>=1.24.0`, `pandas>=2.0.0`, `pillow>=10.0.0`, `scikit-learn>=1.3.0`, `reportlab>=5.0.0`, `matplotlib>=3.10.0`, `pytest>=7.4.0`.

### Código Fuente Oficial
Repositorio GitHub: `https://github.com/ginozza/alzheimer-vit-smri`

### Especificación Algorítmica y Pseudocódigo Formal
*(Consultar el PDF oficial [`docs/reports/Articulo_Final_Seminario_III.pdf`](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/Articulo_Final_Seminario_III.pdf) para la reproducción tipográfica en dos columnas de los Apéndices A, B, C, D y E en las páginas 10 a 18).*
