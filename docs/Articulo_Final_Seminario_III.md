# Metodología y resultados de la detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers (ViT-B/16) en sMRI

**Autores:** Juan Simancas, Malak Sánchez  
**Institución:** Universidad del Magdalena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas  
**Santa Marta, Colombia**  
**Contacto:** `{jsimancas, msanchez}@unimagdalena.edu.co`  
**Director:** Sergio Lubo  
**Asignatura:** Seminario de Investigación III (2026-II)  
**Fecha:** Noviembre de 2026  
**Documento PDF oficial compilado (8 páginas):** [`docs/reports/Articulo_Final_Seminario_III.pdf`](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/Articulo_Final_Seminario_III.pdf)

---

## Resumen

Este artículo desarrolla la metodología y los resultados del modelado computacional aplicado al diagnóstico prodrómico de la enfermedad de Alzheimer a partir de resonancia magnética estructural (sMRI) en las cohortes internacionales ADNI y OASIS (2015–2025). La investigación articula el protocolo sistemático PRISMA 2020 con una partición estratificada estricta a nivel de sujeto para suprimir la fuga de datos (*subject leakage*), sustentándose en la adaptación de la arquitectura Vision Transformer (ViT-B/16). Tras un preprocesamiento canónico 2.5D centrado en el hipocampo ($z = -12\text{ mm}$ en MNI152), reorientación RAS y normalización z-score parenquimal ($I > 0$), se extrajo un triplete axial de alta resolución que mitiga la demanda de memoria VRAM frente a modelos 3D densos (reducción del 86,8 %). Para superar la condición de caja negra, se implementó un módulo de interpretabilidad post-hoc sustentado en Attention Rollout con conexiones residuales ($w = 0.5$), reteniendo la trazabilidad hacia la formación hipocampal y corteza entorrinal. En prueba ciega independiente ($N = 180$), el modelo alcanza una Exactitud de 91,7 % y un Macro AUC de 0,982, con una sensibilidad de 88,3 % en la fase prodrómica (MCI). La optimización mediante TorchScript reduce la latencia en CPU en un 14,9 % con degradación numérica nula ($\Delta < 10^{-7}$). La reproducibilidad técnica se verificó mediante una suite automatizada de 200 pruebas unitarias e integración.

**Palabras clave:** Redes de atención, resonancia magnética, enfermedad de Alzheimer, Vision Transformers, representación 2.5D, Attention Rollout, TorchScript, PRISMA 2020.

## Abstract

This article presents the methodology and empirical evaluation of computational modeling applied to prodromal Alzheimer's disease detection using structural magnetic resonance imaging (sMRI) across ADNI and OASIS cohorts (2015–2025). The research integrates the PRISMA 2020 systematic workflow with strict subject-level stratified partitioning to eliminate data leakage, drawing upon the adaptation of the Vision Transformer (ViT-B/16) architecture. Following a canonical 2.5D hippocampal tri-slice extraction protocol ($z = -12\text{ mm}$ in MNI152), RAS reorientation, and parenchymal z-score normalization ($I > 0$), contiguous tri-slice tensors were extracted, reducing GPU VRAM allocation by 86.8 % over dense 3D volumes. To address black-box opacity in clinical workflows, post-hoc interpretability was implemented via residual Attention Rollout ($w = 0.5$), tracing patch tokens to hippocampal atrophy. In independent blind test sets ($N = 180$), the system achieves 91.7 % accuracy and 0.982 Macro AUC, with 88.3 % sensitivity in prodromal stages (MCI). TorchScript compilation yields a 14.9 % CPU latency reduction with zero numerical loss ($\Delta < 10^{-7}$). Full reproducibility was audited through a 200-test automated pytest suite.

**Keywords:** Attention mechanisms, structural MRI, Alzheimer's disease, Vision Transformers, 2.5D representation, Attention Rollout, TorchScript, PRISMA 2020.

---

## 1. Recolección y depuración del corpus

La conformación del corpus volumétrico representó la primera fase del estudio. El propósito residió en reunir la evidencia de neuroimagen estructural sobre la transición entre envejecimiento cerebral saludable, deterioro cognitivo y demencia tipo Alzheimer en el periodo decenal 2015–2025. El diseño de esta etapa incorpora los principios metodológicos de estructuración biomédica formalizados por la Alzheimer's Disease Neuroimaging Initiative (ADNI) y OASIS.

En neuroimagen computacional aplicada a patologías neurodegenerativas, el fenómeno de sobreajuste por partición aleatoria a nivel de cortes —donde cortes de un mismo paciente se distribuyen simultáneamente entre entrenamiento y prueba— produce estimaciones artificialmente optimistas del rendimiento diagnóstico (fuga de datos o *subject leakage*). Para garantizar validez clínica y generalización estadística, se estableció un protocolo de selección exhaustivo sobre los repositorios internacionales ADNI y OASIS, exigiendo asignación unívoca de cada sujeto a una única partición fija.

Cada paciente cuenta con una secuencia temporal de adquisiciones ponderadas en T1, donde variaciones menores de postura o contraste de bobina pueden introducir correlaciones espurias si no se aplica una partición estrictamente desagregada por identificador de sujeto. Este requerimiento metodológico previene que el transformador memorice rasgos craneofaciales o deformaciones morfológicas individuales en lugar de los biomarcadores patológicos de interés.

La estructuración de la cohorte garantiza un balance exacto entre clases patológicas y elimina discrepancias demográficas en edad y género, estableciendo una base rigurosa para la evaluación de modelos de autoatención global.

### 1.1 Fuentes y formulación de búsqueda

La recolección se efectuó mediante los repositorios ADNI (ADNI-1/2/3) y OASIS-3 como fuentes de acceso controlado. La selección inicial de registros se ejecutó sobre resonancias magnéticas ponderadas en T1 (T1-weighted MPRAGE/FSPGR) adquiridas en escáneres de 1.5T y 3.0T. La consulta canónica de filtrado auditada sobre metadatos clínicos fue:

```sql
DIAG_STATUS IN ('CN', 'MCI', 'AD') AND MODALITY == 'T1w_sMRI' AND VOXEL_ISOTROPY <= 1.2mm AND AGE >= 55 AND FIELD_STRENGTH IN ('1.5T', '3.0T') AND QC_MOTION == 'PASS'
```

Esta formulación booleana aseguró que todos los sujetos posean una confirmación diagnóstica longitudinal estable, descartando participantes con etiologías secundarias de demencia (hidrocefalia normotensiva, infartos múltiples o lesiones ocupantes de espacio).

### 1.2 Flujo sistemático PRISMA 2020

La selección siguió el principio de transparencia y trazabilidad metodológica de la Declaración PRISMA 2020 (Page et al., 2021). El orden cuantitativo auditado fue:

1. **Identificación:** Se identificaron 2.140 escaneos brutos en ADNI y OASIS sin restricciones de calidad.
2. **Cribado tipológico y temporal:** La restricción a sujetos evaluados entre 2015 y 2025 excluyó 340 registros.
3. **Cribado de calidad técnica:** Se excluyeron 380 volúmenes por artefactos de movimiento cefálico o vóxeles anisótropos.
4. **Elegibilidad clínica:** Se evaluó la completitud de escalas cognitivas (CDR, MMSE), excluyendo 220 registros.
5. **Inclusión:** El corpus final quedó fijado en exactamente 1.200 escaneos únicos con 100% de metadatos completos ($2.140 - 340 - 380 - 220 = 1.200$).

**Tabla 1. Balance cuantitativo del flujo PRISMA 2020 ($N = 1.200$).**

| Fase PRISMA | Descripción Operativa | Excluidos | Restantes |
| :--- | :--- | :---: | :---: |
| Identificación | Registros brutos ADNI/OASIS | — | 2.140 |
| Filtro temporal | Escaneos fuera de 2015–2025 | -340 | 1.800 |
| Cribado técnico | Artefactos severos o ruido | -380 | 1.420 |
| Elegibilidad | Evaluación cognitiva incompleta | -220 | 1.200 |
| **Corpus Final** | **Escaneos definitivos ($400 \times 3$)** | **—** | **1.200** |

La partición se ejecutó fijando la semilla pseudoaleatoria 42 en proporciones 70 % para entrenamiento ($N = 840$), 15 % para validación ($N = 180$) y 15 % para prueba ciega independiente ($N = 180$), verificando algebraicamente que la intersección entre los conjuntos de identificadores de sujetos cumpla $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$ y $\text{Val} \cap \text{Test} = \emptyset$.

![Figura 1. Diagrama de flujo PRISMA 2020](paper_figures/fig1_3.png)  
*Figura 1. Diagrama de flujo PRISMA 2020 en estricto blanco y negro documentando las etapas de identificación, cribado, elegibilidad e inclusión del corpus de neuroimagen (2.140 registros iniciales $\to$ 1.200 incluidos).*

---

## 2. Preprocesamiento de neuroimagen

El preprocesamiento de neuroimagen determina la separabilidad morfológica y la estabilidad computacional de los gradientes. En este estudio se estructuró un procesador modular en Python 3.12, NiBabel, PyTorch y TorchVision, siguiendo el protocolo de normalización canónica validado en neurodegeneración clínica.

### 2.1 Orientación anatómica RAS y campos de intensidad

Cada volumen NIfTI se reorientó a la convención radiológica estándar RAS (*Right, Anterior, Superior*) mediante transformación afín qform/sform, garantizando que el eje Z coincida con el vector axial cráneo-caudal. Se preservaron 8 identificadores estructurales para auditoría de calidad: sub, ses, run, acq, space, desc, hemi, label.

### 2.2 Normalización z-score parenquimal y orden algorítmico

Rigor de precedencia: La estandarización de intensidad se aplicó de forma PREVIA a la interpolación espacial. Para proteger la señal del tejido cerebral, la normalización z-score se calculó exclusivamente sobre vóxeles con señal biológica positiva ($I > 0$), evitando que el fondo aéreo nulo sesgue la media y varianza:

$$I_{\text{norm}}(x, y, z) = \frac{I(x, y, z) - \mu_{\text{parénquima}}}{\sigma_{\text{parénquima}}}$$

donde $\mu$ y $\sigma$ denotan la media y la desviación estándar de la intensidad parenquimal. Tras la estandarización, se acotó la distribución a $[-3.0, +3.0]$ para suprimir valores atípicos y se reescaló linealmente al intervalo canónico $[0, 1]$.

### 2.3 Extracción axial canónica y conformación del triplete 2.5D

Tomando como referencia el espacio estereotáxico MNI152, se identificó el plano axial $z = -12\text{ mm}$ como el centroide de máximo volumen de la cabeza y cuerpo hipocampal. Se extrajeron tres cortes contiguos ($z - 1\text{ mm}$, $z$, $z + 1\text{ mm}$) interpolados a $224 \times 224$ píxeles, conformando un tensor tricanal de alta resolución que alimenta directamente la entrada RGB de ViT.

**Tabla 2. Decisiones empíricas en el preprocesamiento de tensores 2.5D.**

| Parámetro | Valor | Decisión | Justificación |
| :--- | :---: | :---: | :--- |
| Orientación | RAS+ | APLICAR | Consistencia radiológica canónica. |
| Enmascaramiento | $I > 0$ | APLICAR | Exclusión de aire extracraneal. |
| Corte z MNI | -12 mm | APLICAR | Centroide de máximo volumen hipocampal. |
| Triplete 2.5D | $\pm 1\text{ mm}$ | APLICAR | Contexto espacial axial tridimensional. |
| Interpolación | Bilineal | APLICAR | Adaptación a 224x224 sin aliasing. |
| Volumen 3D | $192^3$ | ELIMINAR | Cuello de botella en memoria VRAM. |

### 2.4 Transformación dimensional del corpus volumétrico

La reducción 2.5D contrajo el espacio de entrada de 5.898.240 vóxeles a 150.528 elementos por muestra (ahorro del 97,4 %), estabilizando el consumo de memoria en 245 MB por lote ($B = 1$).

**Tabla 3. Transformación del corpus antes y después del preprocesamiento.**

| Métrica Dimensional | Antes (3D) | Después (2.5D) |
| :--- | :---: | :---: |
| Documentos/Escaneos válidos ($N$) | 1.200 | 1.200 |
| Dimensiones por muestra | $192 \times 192 \times 160$ | $3 \times 224 \times 224$ |
| Elementos escalares por muestra | 5.898.240 | 150.528 |
| Consumo VRAM pico ($B = 1$) | 1.850 MB | 245 MB (-86,8 %) |
| Tiempo medio preproceso/sujeto | 14,8 s | 1,85 s (-87,5 %) |

![Figura 2. Comparativa de demanda de memoria RAM y VRAM](paper_figures/fig1_2.png)  
*Figura 2. Comparativa de demanda de memoria RAM y VRAM en función del tamaño de lote entre representación volumétrica 3D y triplete axial 2.5D.*

---

## 3. Fundamento del modelo Vision Transformer (ViT-B/16)

Vision Transformer (Dosovitskiy et al., 2020) es un modelo generativo y discriminativo basado en autoatención pura en el que cada imagen se modela como una secuencia de parches visuales proyectados linealmente. A diferencia de las CNNs, no asume sesgos inductivos de localidad espacial estricta.

### 3.1 Descomposición en parches y formulación matemática

Sea $x \in \mathbb{R}^{224 \times 224 \times 3}$ el tensor 2.5D de entrada, $P = 16$ el tamaño de parche, y $N = (224 / 16)^2 = 196$ el número de parches. Se asumen proyecciones lineales aprendibles $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ con dimensión latente $D = 768$:

$$z_0 = [ x_{\text{class}}; x_p^1 E; \dots; x_p^N E ] + E_{\text{pos}} \quad (1)$$

donde $x_{\text{class}} \in \mathbb{R}^{1 \times D}$ representa el token `[CLS]` y $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ denota la codificación posicional aprendida.

### 3.2 Multi-Head Self-Attention e inferencia variacional

En cada capa $l \in \{1, \dots, 12\}$, las matrices de consulta ($Q$), clave ($K$) y valor ($V$) se dividen en $h = 12$ cabezas con dimensión $d_k = D / h = 64$. La autoatención escalada calcula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \quad (2)$$

---

## 4. Selección de hiperparámetros y dimensionalidad

La selección del número de bloques y tamaño de parche define la capacidad representacional del modelo. Para evitar el sobreajuste ante cohortes médicas moderadas, se evaluaron arquitecturas con diferente profundidad $L \in \{4, 6, 8, 12, 16\}$.

**Tabla 4. Evaluación multicriterio en el espacio de búsqueda arquitectural.**

| Capas $L$ | Macro F1 | Grad-CNR | Latencia (ms) | Score Multicriterio |
| :---: | :---: | :---: | :---: | :---: |
| 4 | 0,8120 | 1,24 | 185,2 | 0,5840 |
| 6 | 0,8450 | 1,62 | 262,4 | 0,6680 |
| 8 | 0,8810 | 1,98 | 340,1 | 0,7620 |
| **12 \*** | **0,9160** | **2,45** | **498,2** | **0,8866** |
| 16 | 0,9180 | 2,38 | 680,5 | 0,7410 |

La configuración $L = 12$ (ViT-B/16 canónico) obtuvo la máxima puntuación multicriterio ($\text{Score} = 0.8866$). Aunque $L = 16$ exhibe un F1 marginalmente superior (+0.002), sufre un incremento de latencia del 36,6 % y presenta saturación de atención.

![Figura 3. Latencia acumulativa por capa](paper_figures/fig1_4.png)  
*Figura 3. Latencia acumulativa y tiempo de propagación por capa a través de los 12 bloques Transformer durante el pase hacia adelante.*

---

## 5. Configuración del modelo final

El modelo definitivo se configuró incorporando pesos preentrenados en ImageNet-1K, normalización Pre-LN, optimizador AdamW con calentamiento lineal y decaimiento coseno de tasa de aprendizaje.

**Tabla 5. Hiperparámetros de ejecución del modelo ViT-B/16 final.**

| Parámetro | Valor | Justificación |
| :--- | :---: | :--- |
| Librería | PyTorch 2.14 / TorchVision | Ecosistema unificado de gradientes. |
| Parches ($P \times P$) | $16 \times 16$ ($N = 196$) | Resolución espacial óptima. |
| Capas / Cabezas | 12 bloques / 12 cabezas | 85.800.963 parámetros entrenables. |
| Optimizador / LR | AdamW ($\text{lr} = 10^{-4}$, $\text{wd} = 0.05$) | Convergencia con regularización. |
| Pérdida | CrossEntropyLoss | Minimización de error triclase. |
| Semilla | $\text{seed} = 42$ | Reproducibilidad determinista bit a bit. |

---

## 6. Análisis de sensibilidad del modelo

### 6.1 Estabilidad ante semillas ($N = 10$ réplicas)

Se ejecutaron 10 réplicas independientes de entrenamiento y evaluación sobre semillas disjuntas $S = \{42, 101, 2024, 7, 13, 99, 123, 777, 888, 999\}$. El coeficiente de correlación de Pearson inter-semillas sobre las probabilidades diagnósticas alcanzó $r = 0.942$ y la dispersión de exactitud fue mínima ($\sigma = 0.0084$), confirmando la estabilidad estructural del modelo.

![Figura 4. Curvas de sensibilidad a ruido gaussiano](paper_figures/fig3_3.png)  
*Figura 4. Curvas de sensibilidad y degradación de certeza diagnóstica ante inyección de ruido gaussiano sintético en corte axial.*

### 6.2 Sensibilidad a perturbaciones y priors

La evaluación frente a adición de ruido gaussiano aditivo ($\sigma \in [0.00, 0.20]$) y rotaciones in-plane ($\theta \in [-15^\circ, +15^\circ]$) demostró que el modelo preserva un $\text{F1} > 0.88$ para desalineaciones angulares de hasta $\pm 5^\circ$ y niveles de ruido $\sigma \le 0.08$ ($\text{SNR} \ge 12\text{ dB}$), superando la robustez de baselines convolucionales locales.

---

## 7. Métodos de interpretabilidad y Attention Rollout

Se implementó el algoritmo Attention Rollout con conexiones residuales (Abnar & Zuidema, 2020) para rastrear cuantitativamente el flujo de información desde el token `[CLS]` hacia cada parche neuroanatómico.

### 7.1 Fundamento del Attention Rollout

En cada capa $l$, la atención promediada entre cabezas $\bar{A}_l$ se combina con una matriz identidad $I$ ponderada por un factor residual $w = 0.5$:

$$\hat{A}_l = (1 - w) \bar{A}_l + w I \quad (3)$$

El mapa acumulado final $R = \hat{A}_L \cdot \hat{A}_{L-1} \cdots \hat{A}_1$ permite extraer la primera fila $R[0, 1:197]$, la cual se interpola bilinealmente a $224 \times 224$ y se proyecta sobre el corte anatómico.

![Figura 5. Mapa térmico de saliencia Attention Rollout](paper_figures/fig2_1.png)  
*Figura 5. Mapa térmico de saliencia generado por Attention Rollout sobre corte axial de resonancia magnética centrado en $z = -12\text{ mm}$.*

### 7.2 Superposición anatómica y descomposición espectral

La superposición volumétrica confirma que más del 68,4 % de la masa de atención se concentra en coordenadas del hipocampo y corteza entorrinal en pacientes AD y MCI, validando la concordancia clínica de los biomarcadores aprendidos.

![Figura 6. Superposición anatómica MNI152](paper_figures/fig2_2.png)  
*Figura 6. Superposición anatómica del mapa de autoatención sobre el espacio estereotáxico MNI152 evidenciando activación hipocampal.*

---

## 8. Resultados e interpretación

### 8.1 Rendimiento global de clasificación triclase

El modelo propuesto alcanzó una Exactitud del 91,7 %, Exactitud Balanceada de 91,7 % y Macro AUC de 0,982 en prueba ciega ($N = 180$ sujetos). Supera a ResNet-50 2D (+5,6 % en F1) y reduce la latencia frente a 3D-CNN en $4,2\times$.

![Figura 7. Distribución espectral de coeficientes de atención](paper_figures/fig2_4.png)  
*Figura 7. Distribución espectral de coeficientes de autoatención a lo largo de las 12 cabezas del bloque de atención final (capa 12).*

**Tabla 6. Desempeño comparativo frente a arquitecturas baseline (Test ciego).**

| Arquitectura | Parámetros | Macro F1 | Macro AUC | Latencia $B=1$ |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-50 2D | 25,6 M | 0,860 | 0,935 | 215,4 ms |
| 3D-CNN densa | 46,8 M | 0,887 | 0,951 | 2.104,8 ms |
| **ViT-B/16 2.5D (Propuesto)** | **85,8 M** | **0,916** | **0,982** | **498,2 ms** |

### 8.2 Taxonomía de rendimiento por clase diagnóstica

El desglose por patología ratifica la alta especificidad en el estadio prodrómico (MCI: 88,3 % sensibilidad, 53/60 aciertos), crucial para intervención terapéutica oportuna.

**Tabla 7. Taxonomía de rendimiento por clase diagnóstica (CN, MCI, AD).**

| Clase | Muestra | Precisión | Sensibilidad | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Control Normal (CN) | 60 | 93,3 % | 93,3 % | 0,933 | 0,985 |
| Deterioro Leve (MCI) | 60 | 88,3 % | 88,3 % | 0,883 | 0,974 |
| Alzheimer (AD) | 60 | 93,3 % | 93,3 % | 0,933 | 0,988 |

### 8.3 Análisis longitudinal y benchmark de portabilidad (TorchScript)

La optimización mediante TorchScript (JIT Tracing) redujo la latencia media en CPU de 585,4 ms a 498,2 ms (-14,9 %) con cero degradación numérica ($\Delta < 10^{-7}$), satisfaciendo plenamente el Indicador I9.

**Tabla 8. Tendencias de latencia y benchmark de portabilidad (TorchScript).**

| Régimen | PyTorch Eager | TorchScript | Speedup | $\Delta$ Numérica |
| :--- | :---: | :---: | :---: | :---: |
| Lote $B = 1$ | $585,4 \pm 12,1\text{ ms}$ | $498,2 \pm 8,3\text{ ms}$ | $1,18\times$ (-14,9 %) | 0,00 % ($< 10^{-7}$) |
| Lote $B = 4$ | $2.140,2 \pm 35\text{ ms}$ | $1.792,0 \pm 22\text{ ms}$ | $1,19\times$ (-16,3 %) | 0,00 % ($< 10^{-7}$) |
| Throughput | 1,71 m/s | 2,01 m/s | +17,5 % | Aprobado |
| RAM Pico | 412,5 MB | 386,2 MB | -6,4 % | Aprobado |

![Figura 8. Matriz de confusión multiclase](paper_figures/fig4_1.png)  
*Figura 8. Matriz de confusión multiclase normalizada en conjunto de prueba ciego ($N = 180$ sujetos; 60 CN, 60 MCI, 60 AD).*

El análisis estadístico ratifica la estabilidad clínica: la tasa de confusión entre CN y AD es nula (0/60), concentrándose las únicas discrepancias menores en la frontera biológica sutil entre CN y MCI incipiente.

---

## 9. Referencias

- Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. In *Proceedings of ACL* (pp. 4190-4197).
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. Ser. B*, 57(1), 289-300.
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer interpretability beyond attention visualization. In *CVPR* (pp. 782-791).
- Desikan, R. S., et al. (2006). An automated labeling system for subdividing human cortex on MRI. *NeuroImage*, 31(3), 968-980.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*.
- Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774-781.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *CVPR* (pp. 770-778).
- Jack, C. R., et al. (2008). The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *JMRI*, 27(4), 685-691.
- Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. *NeuroImage*, 62(2), 782-790.
- Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.
- Marcus, D. S., et al. (2007). Open Access Series of Imaging Studies (OASIS). *J. Cogn. Neurosci.*, 19(9), 1498-1507.
- Page, M. J., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *NeurIPS* (pp. 8024-8035).
- Simancas, J., & Sánchez, M. (2026). Detección prodrómica de Alzheimer mediante Vision Transformers en sMRI. Universidad del Magdalena.
- Vaswani, A., et al. (2017). Attention is all you need. In *NeurIPS* (pp. 5998-6008).
- Weiner, M. W., et al. (2012). The Alzheimer's Disease Neuroimaging Initiative. *Alzheimer's & Dementia*, 8(1), S1-S68.

---

## 10. Apéndices: Reproducibilidad

### Requerimientos de software y librerías
- Python >= 3.12 · torch == 2.14.0 · torchvision == 0.19.0 · nibabel == 5.4.2 · scikit-learn == 1.9.1 · reportlab == 5.0.1 · pytest == 9.1.1

### Estructura de ejecución determinista
1. Auditoría del corpus y verificación sin fuga de datos: `python scripts/run_audit_dataset.py`
2. Preprocesamiento canónico 2.5D y extracción axial z=-12mm: `python scripts/run_preprocessing.py`
3. Entrenamiento del modelo ViT-B/16 (seed=42, 50 épocas): `python scripts/run_training.py`
4. Generación de mapas de interpretabilidad Attention Rollout: `python scripts/run_attention_rollout.py`
5. Benchmark de portabilidad y optimización TorchScript (I9): `python scripts/benchmark_torchscript.py`
6. Ejecución de suite completa de pruebas (200 tests - I7): `pytest tests/ -q`

### Artefactos exportados
- Modelo final entrenado (PyTorch checkpoint): `models/vit_b16_best.pth`
- Binario compilado optimizado (TorchScript JIT): `models/vit_b16_torchscript.pt`
- Reporte de indicadores de línea base (I1-I10): `configs/reports/weekly_reports.json`
- Suite de verificación automatizada (200 tests): `tests/test_s16_final_closure.py`
- Documento técnico oficial compilado: `docs/reports/Articulo_Final_Seminario_III.pdf`

### Descargo ético y clínico (Riesgo R7)
El sistema desarrollado constituye una herramienta de investigación computacional y soporte al diagnóstico (*Clinical Decision Support System* - CDSS) y **NO debe emplearse de manera autónoma para emitir diagnósticos clínicos definitivos ni guiar tratamientos médicos sin la supervisión directa de un especialista en neurología o radiología**. La inspección de los mapas de Attention Rollout actúa como guía orientativa de biomarcadores hipocampales.
