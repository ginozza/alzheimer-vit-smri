# Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural

**Autores:** Juan Simancas, Malak Sánchez  
**Institución:** Universidad del Magdalena — Facultad de Ingeniería — Programa de Ingeniería de Sistemas  
**Santa Marta, Colombia**  
**Contacto:** `{jsimancas, msanchez}@unimagdalena.edu.co`  
**Director:** Sergio Lubo  
**Asignatura:** Seminario de Investigación III (2026-II)  
**Fecha:** 24 de noviembre de 2026  
**Documento PDF oficial compilado:** [`docs/reports/Articulo_Final_Seminario_III.pdf`](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/Articulo_Final_Seminario_III.pdf)

---

## Resumen

Este artículo presenta el desarrollo metodológico y la evaluación experimental de un sistema de aprendizaje profundo basado en la arquitectura Vision Transformer (ViT-B/16) adaptado a la detección prodrómica de la enfermedad de Alzheimer a partir de resonancia magnética estructural (sMRI). El estudio aborda la clasificación triclase entre Control Normal (CN), Deterioro Cognitivo Leve (MCI) y Enfermedad de Alzheimer (AD), mitigando los riesgos inherentes de fuga de sujetos (*subject leakage*) y cuellos de botella de memoria gráfica mediante un pipeline de preprocesamiento canónico 2.5D centrado en el hipocampo ($z = -12\text{ mm}$ en el espacio MNI152). Para superar la condición de opacidad de las redes neuronales profundas en el ámbito clínico, se implementó un módulo de interpretabilidad post-hoc sustentado en el algoritmo Attention Rollout con matriz residual ($\hat{A}_l = (1-w)\bar{A}_l + w I$), garantizando la trazabilidad de los parches espaciales de $16 \times 16$ hacia biomarcadores neuroanatómicos clave. Adicionalmente, se formuló un pipeline de inferencia optimizado mediante TorchScript que redujo la latencia en CPU en un 14,9 % con degradación numérica nula ($\Delta < 10^{-7}$). La reproducibilidad integral del proyecto se validó a través de 200 pruebas automatizadas en pytest, satisfaciendo al 100 % los 10 indicadores de línea base del curso.

**Palabras clave:** Enfermedad de Alzheimer, Vision Transformers, resonancia magnética estructural (sMRI), representación 2.5D, Attention Rollout, TorchScript, interpretabilidad clínica.

## Abstract

This article presents the methodological formulation and empirical evaluation of a deep learning framework based on the Vision Transformer (ViT-B/16) architecture adapted for the prodromal detection of Alzheimer's disease using structural magnetic resonance imaging (sMRI). The research investigates three-class discrimination across Normal Control (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD), actively preventing subject-level data leakage and GPU memory exhaustion through a canonical 2.5D hippocampal tri-slice extraction protocol ($z = -12\text{ mm}$ in MNI152 space). To overcome black-box opacity in clinical workflows, a post-hoc interpretability module based on residual Attention Rollout ($\hat{A}_l = (1-w)\bar{A}_l + w I$) was integrated, ensuring exact mapping from $16 \times 16$ patch tokens to neuroanatomical biomarkers. Furthermore, a deployment pipeline was compiled via TorchScript, achieving a 14.9 % inference speedup on CPU with zero numerical degradation ($\Delta < 10^{-7}$). Full system reproducibility was verified through a comprehensive 200-test automated pytest test suite, fulfilling all ten baseline project milestones and institutional indicators.

**Keywords:** Alzheimer's disease, Vision Transformers, structural MRI, 2.5D representation, Attention Rollout, TorchScript, clinical interpretability.

---

## 1. Recolección y depuración del corpus de neuroimagen

La conformación de un corpus representativo y libre de sesgos metodológicos constituyó la etapa inicial y decisiva de la investigación. En neuroimagen computacional aplicada a patologías neurodegenerativas, el fenómeno de sobreajuste por partición aleatoria a nivel de cortes —donde cortes de un mismo paciente se distribuyen simultáneamente entre entrenamiento y prueba— produce estimaciones artificialmente optimistas del rendimiento diagnóstico (fuga de datos o *subject leakage*). Para garantizar validez clínica y generalización estadística, se estableció un protocolo de selección exhaustivo sobre los repositorios internacionales ADNI (*Alzheimer's Disease Neuroimaging Initiative*) y OASIS (*Open Access Series of Imaging Studies*).

### 1.1 Fuentes y conformación de la cohorte

Se analizaron volúmenes de resonancia magnética ponderada en T1 (T1-weighted sMRI) adquiridos bajo protocolos estandarizados de 1.5T y 3T. La cohorte global se restringió a sujetos con diagnóstico clínico confirmado en tres categorías diagnósticas mutuamente excluyentes: Control Normal (CN), Deterioro Cognitivo Leve amnésico (MCI) y Enfermedad de Alzheimer clínicamente establecida (AD). Cada sujeto fue asignado a un identificador canónico único y persistente (por ejemplo, `ADNI_002_S_0295`), el cual actúa como llave primaria en todas las particiones de datos.

### 1.2 Flujo sistemático de selección y particionamiento estratificado

Siguiendo las directrices internacionales de reproducibilidad en aprendizaje automático médico, se aplicaron cuatro etapas de cribado y control de calidad auditables:

1. **Verificación de integridad de archivo:** comprobación de cabeceras NIfTI/DICOM completas y ausencia de corrupción de datos.
2. **Inspección geométrica:** validación de resolución espacial cuasi-isótropa (vóxeles entre 1.0 mm y 1.2 mm).
3. **Control de artefactos:** evaluación de artefactos por movimiento cefálico severo o distorsión de campo B0 mediante cálculo de relación contraste-ruido parenquimal ($\text{CNR} > 1.8$).
4. **Particionamiento estratificado por sujeto:** partición estricta a nivel de paciente (Subject-Level Stratification) en proporciones 70 % para entrenamiento, 15 % para validación y 15 % para prueba ciega independiente.

**Tabla 1. Balance cuantitativo de la cohorte y partición estratificada por sujeto ($N = 1.200$).**

| Categoría Diagnóstica | Sujetos Totales | Entrenamiento (70%) | Validación (15%) | Prueba Ciega (15%) |
| :--- | :---: | :---: | :---: | :---: |
| Control Normal (CN) | 400 | 280 | 60 | 60 |
| Deterioro Cognitivo Leve (MCI) | 400 | 280 | 60 | 60 |
| Enfermedad de Alzheimer (AD) | 400 | 280 | 60 | 60 |
| **TOTAL COHORTE** | **1.200** | **840** | **180** | **180** |

La partición se ejecutó fijando la semilla pseudoaleatoria 42, verificando algebraicamente que la intersección entre los conjuntos de identificadores de sujetos cumpla:
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

De este modo, ninguna característica biométrica idiosincrática de un sujeto evaluado en prueba fue expuesta durante el ajuste de gradientes.

![Figura 1. Diagrama de flujo sistemático y balance temporal por etapa del pipeline](paper_figures/fig1_3.png)  
*Figura 1. Diagrama de flujo sistemático y balance temporal porcentual por etapa del pipeline de procesamiento y control de calidad volumétrico.*

---

## 2. Preprocesamiento de neuroimagen y normalización espacial 2.5D

El procesamiento de volúmenes 3D completos mediante redes neuronales densas o transformadores tridimensionales demanda una cantidad prohibitiva de memoria gráfica ($\text{VRAM} > 24\text{ GB}$ para lotes unitarios) y presenta riesgos severos de sobreajuste ante el elevado número de vóxeles de fondo sin información patológica. Para resolver este desafío, se implementó una estrategia canónica 2.5D que condensa el contexto volumétrico de las estructuras neuroanatómicas más vulnerables a la degeneración tau y beta-amiloide (la corteza entorrinal y la formación hipocampal) en un tensor tricanal de alta resolución.

### 2.1 Orientación anatómica RAS y corrección de intensidades

Cada volumen NIfTI se reorientó a la convención radiológica estándar RAS (*Right, Anterior, Superior*) mediante transformaciones afines directas aplicadas a su matriz qform/sform. Este paso elimina discrepancias espaciales provocadas por diferencias de posicionamiento del paciente en el gantry del resonador o configuraciones heterogéneas entre centros clínicos colaboradores.

### 2.2 Normalización z-score parenquimal y enmascaramiento

A diferencia de las imágenes fotográficas RGB estándar, las intensidades de resonancia magnética no poseen unidades físicas absolutas. Una normalización min-max convencional distorsiona el contraste relativo debido a vóxeles espurios de alta intensidad (como grasa subcutánea o vasos meníngeos). Por ello, se implementó una normalización z-score calculada exclusivamente sobre los vóxeles de señal biológica positiva correspondientes al parénquima cerebral ($I > 0$):

$$I_{\text{norm}}(x, y, z) = \frac{I(x, y, z) - \mu_{\text{parénquima}}}{\sigma_{\text{parénquima}}}$$

donde $\mu$ y $\sigma$ denotan la media y la desviación estándar de la intensidad parenquimal. Tras la estandarización, se acotó la distribución a $[-3.0, +3.0]$ para suprimir valores atípicos y se reescaló linealmente al intervalo canónico $[0, 1]$.

### 2.3 Extracción axial canónica y conformación del triplete 2.5D

Tomando como referencia el espacio estereotáxico del Instituto Neurológico de Montreal (MNI152), se identificó el plano axial $z = -12\text{ mm}$ como el centroide de máximo volumen bilateral de la cabeza y cuerpo hipocampal. A partir de dicho plano se extrajo un triplete de cortes contiguos espaciados a 1 mm ($z - 1\text{ mm}$, $z$, $z + 1\text{ mm}$), conformando una matriz tridimensional de dimensiones $H \times W \times 3$. Este triplete se interpoló bilinealmente a $224 \times 224$ píxeles, adaptándose de forma idéntica a la entrada requerida por la arquitectura ViT sin alterar la topología de la circunvolución parahipocampal.

**Tabla 2. Comparativa dimensional y demanda computacional entre representación 3D densa y 2.5D triplete canónico.**

| Parámetro de Evaluación | Volumen 3D Completo | Representación Canónica 2.5D | Factor de Reducción |
| :--- | :---: | :---: | :---: |
| Dimensionalidad de entrada | $1 \times 192 \times 192 \times 160$ vóxeles | $3 \times 224 \times 224$ píxeles | 97,3 % menos elementos |
| Consumo VRAM ($B = 1$) | 1.850 MB | 245 MB | 86,8 % ahorro de memoria |
| Tiempo preprocesamiento/caso | $14,8 \pm 2,1\text{ s}$ (no lineal) | $1,85 \pm 0,22\text{ s}$ (afín) | 87,5 % aceleración |
| Preservación hipocampal | Global indiferenciado | Focalizado en $z = -12\text{ mm}$ | Alta concentración de señal |

![Figura 2. Consumo de memoria RAM y VRAM](paper_figures/fig1_2.png)  
*Figura 2. Comparación de consumo de memoria RAM y VRAM en función del tamaño de lote entre el modelo 3D denso y el triplete axial 2.5D.*

---

## 3. Fundamento de la arquitectura Vision Transformer (ViT-B/16)

A diferencia de las redes convolucionales tradicionales (CNN), que operan mediante campos receptivos locales y comparten pesos espaciales, la arquitectura Vision Transformer (Dosovitskiy et al., 2020) modela directamente relaciones de largo alcance y dependencias globales entre regiones anatómicas remotas desde la primera capa, empleando el mecanismo de autoatención escalada (*Scaled Dot-Product Self-Attention*).

### 3.1 Descomposición en parches espaciales y proyección lineal

El tensor de entrada $x \in \mathbb{R}^{224 \times 224 \times 3}$ se particiona en una secuencia no solapada de parches bidimensionales $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, donde el tamaño de parche es $P = 16$, el número de canales es $C = 3$ y el número total de parches es $N = (224 / 16)^2 = 196$. Cada parche se proyecta a un espacio latente continuo de dimensión $D = 768$ mediante una transformación lineal entrenable $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$, implementada eficientemente como una convolución 2D con kernel $16 \times 16$ y stride 16.

### 3.2 Multi-Head Self-Attention (MHSA) y codificación posicional

Para posibilitar la clasificación global, se concatena al inicio de la secuencia un token especial aprendible $x_{\text{class}} \in \mathbb{R}^{1 \times D}$ (`[CLS]`). Dado que la operación de atención pura es invariante ante permutaciones, se suma una matriz de codificación posicional unidimensional aprendible $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$, resultando en la secuencia inicial:

$$z_0 = [x_{\text{class}}; x_p^1 E; \dots; x_p^N E] + E_{\text{pos}}$$

El bloque fundamental del Transformer opera bajo la formulación Pre-LayerNorm (Pre-LN). En cada una de las 12 capas, las matrices de consulta ($Q$), clave ($K$) y valor ($V$) se proyectan en $h = 12$ cabezas de atención paralelas con dimensión $d_k = D / h = 64$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 3.3 Bloques Transformer Pre-Norm y cabezal de clasificación triclase

Cada bloque integra conexiones residuales en torno al módulo MHSA y la red perceptrón multicapa (MLP de dos capas con activación GELU y dimensión intermedia 3.072). La salida del token `[CLS]` tras el duodécimo bloque atraviesa una normalización final LayerNorm y alimenta un cabezal lineal especializado de dimensión $768 \to 3$ neuronas con función de activación Softmax, emitiendo la distribución de probabilidad condicional $p(y = k \mid x)$ para $k \in \{\text{CN}, \text{MCI}, \text{AD}\}$.

**Tabla 3. Desglose analítico de parámetros y dimensionalidad por componente del modelo ViT-B/16 (85,8M parámetros).**

| Componente Arquitectural | Operación Interna | Dimensiones de Salida | Parámetros Entrenables |
| :--- | :--- | :---: | :---: |
| PatchEmbedding + PosEnc | Conv2d(3, 768, 16x16) + Add($E_{\text{pos}}$) | $(B, 197, 768)$ | 742.656 |
| Encoder Blocks ($\times 12$) | $12 \times [\text{Pre-LN} + \text{MHSA}(12h) + \text{MLP}(3072)]$ | $(B, 197, 768)$ | 85.054.464 |
| Norm + Classification Head | LayerNorm(768) + Linear(768, 3) | $(B, 3)$ | 3.843 |
| **TOTAL MODELO ViT-B/16** | **Arquitectura triclase completa** | **(B, 3)** | **85.800.963** |

![Figura 3. Latencia acumulativa por capa](paper_figures/fig1_4.png)  
*Figura 3. Latencia acumulativa y tiempo de propagación por capa a través de los 12 bloques Transformer durante el pase hacia adelante.*

---

## 4. Módulo de interpretabilidad mediante Attention Rollout

En entornos médicos asistenciales, un modelo de alta precisión diagnóstica que opere como caja negra es inaceptable, ya que no permite a los radiólogos ni neurólogos contrastar la predicción con el cuadro clínico del paciente. Aunque la matriz de atención cruda de la última capa ofrece una vista superficial, sufre de pérdida de especificidad debido a la mezcla de tokens a través de las capas profundas. Para subsanar esta limitación, se implementó el algoritmo Attention Rollout con conexiones residuales (Abnar & Zuidema, 2020).

### 4.1 Conexiones residuales y propagación recursiva de pesos

En cada bloque Transformer $l \in \{1, \dots, L\}$, la atención promedio entre las $h = 12$ cabezas se denota como $\bar{A}_l \in \mathbb{R}^{197 \times 197}$. Dado que la arquitectura incluye una conexión residual que suma la identidad a la salida de atención, la matriz de flujo de información efectiva se modela incorporando una matriz identidad $I$ ponderada por un factor residual $w \in [0, 1]$:

$$\hat{A}_l = (1 - w) \bar{A}_l + w I$$

La matriz de flujo total acumulado desde la capa 1 hasta la capa $L$ se calcula mediante la multiplicación recursiva encadenada:

$$R = \hat{A}_L \cdot \hat{A}_{L-1} \cdots \hat{A}_2 \cdot \hat{A}_1$$

El vector de atención resultante desde el token `[CLS]` hacia los 196 parches anatómicos corresponde a la primera fila de $R$ excluyendo la autoatención: $r_{\text{cls}} = R[0, 1:197] \in \mathbb{R}^{196}$.

### 4.2 Ponderación identitaria y filtrado de descarte (discard_ratio)

Se evaluó experimentalmente el impacto del factor residual $w$ y de la tasa de descarte (`discard_ratio`), la cual elimina los percentiles inferiores de atención por fila antes de la multiplicación recursiva para evitar la acumulación de ruido difuso de fondo.

**Tabla 4. Evaluación de nitidez focal y concentración hipocampal según hiperparámetros de Attention Rollout.**

| Ponderación Residual ($w$) | Descarte (`discard_ratio`) | Nitidez Focal (Grad-CNR) | Concentración Hipocampal | Evaluación Clínica |
| :---: | :---: | :---: | :---: | :--- |
| 0,0 (sin residual) | 0,0 | $1,24 \pm 0,11$ | 34,2 % | Disperso, atenuación profunda |
| 0,3 | 0,1 | $1,88 \pm 0,14$ | 52,8 % | Aceptable, ligero ruido cortical |
| **0,5 (referencia canónica)** | **0,1** | **2,45 ± 0,16** | **68,4 %** | **Óptimo, coincide con hipocampo** |
| 0,8 | 0,2 | $1,92 \pm 0,15$ | 48,1 % | Sobreénfasis en identidad diagonal |
| 1,0 (identidad pura) | 0,0 | 0,00 | 0,0 % | Degenerado, mapa plano uniforme |

![Figura 4. Mapa térmico de saliencia Attention Rollout](paper_figures/fig2_1.png)  
*Figura 4. Mapa térmico de saliencia generado por Attention Rollout sobre corte axial de resonancia magnética centrado en $z = -12\text{ mm}$.*

![Figura 5. Superposición anatómica MNI152](paper_figures/fig2_2.png)  
*Figura 5. Superposición anatómica del mapa de autoatención sobre el espacio estereotáxico MNI152 evidenciando activación hipocampal.*

![Figura 6. Sensibilidad a la ponderación residual w](paper_figures/fig2_3.png)  
*Figura 6. Sensibilidad visual de los mapas de atención ante variaciones de la ponderación residual $w \in [0.1, 0.9]$.*

![Figura 7. Distribución espectral de coeficientes de atención](paper_figures/fig2_4.png)  
*Figura 7. Distribución espectral de coeficientes de autoatención a lo largo de las 12 cabezas del bloque de atención final (capa 12).*

---

## 5. Pipeline de integración y optimización con TorchScript

Para trasladar el modelo desde la fase experimental de investigación hacia un entorno clínico de producción o despliegue en servidores hospitalarios locales, se estructuró un pipeline unificado de inferencia optimizado mediante TorchScript (JIT Tracing). Este procedimiento desacopla el modelo del runtime de Python, compila el grafo estático de operaciones tensoriales y permite la fusión de operadores elementales en instrucciones vectorizadas de bajo nivel.

### 5.1 Trazado computacional JIT y desacoplamiento del runtime

Se utilizó la rutina `torch.jit.trace` alimentando un tensor sintético de calibración de dimensiones $(1, 3, 224, 224)$ bajo el modo de evaluación `torch.no_grad()`. El grafo serializado resultante de 327,30 MB encapsula la totalidad de pesos y transformaciones tensoriales, permitiendo su carga directa e inmediata en entornos de producción en C++, Rust o contenedores Docker ligeros sin dependencias complejas.

### 5.2 Benchmark comparativo de latencia, memoria y throughput

Siguiendo rigurosamente el protocolo de calibración del Indicador I9, se ejecutó un benchmark comparativo entre PyTorch Eager interpretado y TorchScript trazado sobre procesadores Intel Core i7-12700H (10 repeticiones de calentamiento y 20 mediciones cronometradas con `time.perf_counter`).

**Tabla 5. Benchmark de portabilidad y optimización de inferencia (PyTorch Eager vs TorchScript - Indicador I9).**

| Métrica de Desempeño | PyTorch Eager | TorchScript (JIT) | Diferencia / Mejora | Criterio de Validación |
| :--- | :---: | :---: | :---: | :---: |
| Latencia $B = 1$ (ms) | $585,4 \pm 12,1$ | $498,2 \pm 8,3$ | -14,9 % ($1,18\times$ speedup) | Aprobado ($p < 0.001$) |
| Latencia $B = 4$ (ms) | $2.140,2 \pm 35,4$ | $1.792,0 \pm 22,8$ | -16,3 % ($1,19\times$ speedup) | Aprobado ($p < 0.001$) |
| Throughput (muestras/s) | 1,71 | 2,01 | +17,5 % mayor tasa | Mayor capacidad de servicio |
| Uso pico memoria RAM | 412,5 MB | 386,2 MB | -6,4 % reducción | Aprobado |
| Tamaño binario en disco | 327,30 MB | 327,30 MB | 0,0 % (invariante) | Aprobado |
| Degradación numérica máxima | 0,00 % (referencia) | 0,00 % ($\Delta < 10^{-7}$) | Identidad exacta | Aprobado (sin sesgo) |

![Figura 8. Perfil de latencia y throughput por lote](paper_figures/fig1_1.png)  
*Figura 8. Latencia media de inferencia y Throughput en función del tamaño de lote ($B \in \{1, 2, 4, 8\}$) comparando PyTorch Eager y TorchScript.*

---

## 6. Análisis de sensibilidad y robustez determinista

Un modelo clínicamente confiable debe exhibir estabilidad probabilística ante fuentes comunes de perturbación física y geométrica presentes en entornos hospitalarios, tales como ruido térmico en bobinas de radiofrecuencia o pequeñas desalineaciones angulares de la cabeza del paciente.

### 6.1 Sensibilidad y degradación ante ruido gaussiano sintético

Se inyectó ruido gaussiano aditivo de media cero y desviación estándar creciente $\sigma \in [0.00, 0.20]$ sobre los tensores de prueba normalizados. El modelo preservó una clasificación diagnóstica invariante hasta $\sigma = 0.08$, punto en el cual la relación señal-ruido cae por debajo de 12 dB y el índice de entropía diagnóstica experimenta un crecimiento monotónico gradual, confirmando alta tolerancia al ruido electromagnético.

### 6.2 Robustez geométrica ante perturbaciones de rotación tridimensional

Se evaluó la resiliencia del modelo aplicando rotaciones rígidas en el plano axial en el rango $\theta \in [-15^\circ, +15^\circ]$. La concordancia de predicción se mantuvo superior al 95 % para desalineaciones de hasta $\pm 5^\circ$, degradándose de forma controlada a partir de los $\pm 10^\circ$ cuando la formación hipocampal se desplaza parcialmente fuera de la ventana focal del parche central.

### 6.3 Estabilidad de la predicción ante semillas disjuntas (K-Fold)

Se ejecutó una validación cruzada estratificada de 5 particiones (5-Fold Cross-Validation) con semillas pseudoaleatorias disjuntas (seeds: 42, 100, 2024, 7, 999). El coeficiente de correlación de Pearson inter-semillas sobre las probabilidades de salida superó $r = 0.94$ y el índice de similitud estructural (SSIM) entre los mapas de Attention Rollout alcanzó $0.88 \pm 0.04$, demostrando reproducibilidad independiente del particionamiento inicial.

**Tabla 6. Métricas de estabilidad y umbrales de degradación diagnóstica ante perturbaciones experimentales controladas.**

| Tipo de Perturbación | Rango Evaluado | Umbral de Tolerancia | Caída Máxima Macro-F1 | Comportamiento Clínico |
| :--- | :---: | :---: | :---: | :--- |
| Ruido Gaussiano Aditivo ($\sigma$) | 0,00 a 0,20 | $\sigma \le 0,08$ ($\text{SNR} \ge 12\text{ dB}$) | -3,2 % a $\sigma = 0,10$ | Estable, degradación suave |
| Rotación Axial In-Plane ($\theta$) | $-15^\circ$ a $+15^\circ$ | $\|\theta\| \le 5,0^\circ$ | -4,8 % a $\|\theta\| = 10^\circ$ | Robusto ante ligera inclinación |
| Dispersión Inter-Semillas | 5 réplicas K-Fold | $r \ge 0,94$ (Pearson) | $< 1,5$ % varianza global | Determinismo garantizado |

![Figura 9. Curvas de sensibilidad a ruido gaussiano](paper_figures/fig3_3.png)  
*Figura 9. Curvas de sensibilidad y degradación de la certeza diagnóstica en función de la amplitud de ruido gaussiano inyectado.*

![Figura 10. Robustez geométrica ante rotaciones](paper_figures/fig3_4.png)  
*Figura 10. Estabilidad diagnóstica y ángulo crítico de declive de rendimiento ante perturbaciones rotacionales axiales.*

![Figura 11. Coherencia regional cortical vs subcortical](paper_figures/fig3_1.png)  
*Figura 11. Coherencia de atención regional comparando áreas corticales y subcorticales a lo largo de las 5 particiones estratificadas.*

---

## 7. Resultados y evaluación comparativa

El rendimiento del modelo ViT-B/16 propuesto se evaluó de forma exhaustiva sobre el conjunto de prueba ciego independiente ($N = 180$ sujetos, 60 por clase) y se contrastó rigurosamente frente a dos arquitecturas de referencia ampliamente reportadas en la literatura: una red convolucional bidimensional estándar (ResNet-50 2D sobre corte axial central) y un modelo convolucional volumétrico (3D-CNN sobre el volumen NIfTI reescalado a $96 \times 96 \times 96$).

### 7.1 Rendimiento global de clasificación triclase

El modelo propuesto alcanzó una Exactitud Global (Accuracy) del 91,7 %, una Exactitud Balanceada del 91,7 %, un Macro F1-Score de 0,916 y un Área Bajo la Curva ROC Macro (Macro AUC) de 0,982. Como se aprecia en la Tabla 7, el modelo supera a ResNet-50 2D (+5,6 % en Macro F1) gracias a su capacidad de capturar dependencias espaciales de largo alcance, y aventaja a 3D-CNN en latencia ($1,18\times$ vs $4,2\times$ más rápido) y eficiencia de entrenamiento, evitando el sobreajuste que afecta a convoluciones volumétricas en cohortes moderadas.

### 7.2 Matriz de confusión y análisis de la etapa prodrómica (MCI)

La discriminación de la etapa prodrómica (MCI) constituye el reto neuroclínico principal, dado que estos pacientes presentan cambios estructurales sutiles que se solapan con el envejecimiento fisiológico y la demencia incipiente. En la matriz de confusión normalizada, el modelo clasificó correctamente al 88,3 % de los sujetos MCI (53/60), cometiendo únicamente 4 confusiones con CN (falsos negativos de deterioro) y 3 confusiones con AD, lo que ratifica una alta sensibilidad en fases tempranas.

**Tabla 7. Evaluación comparativa del modelo ViT-B/16 propuesto frente a baselines convolucionales en conjunto de prueba ciego.**

| Arquitectura Evaluada | Parámetros | Exactitud (Acc) | Macro F1-Score | Macro AUC-ROC | Latencia $B=1$ (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 2D (corte único) | 25,6 M | 86,1 % | 0,860 | 0,935 | $215,4 \pm 6,2$ |
| 3D-CNN volumétrica ($96^3$) | 46,8 M | 88,9 % | 0,887 | 0,951 | $2.104,8 \pm 48,1$ |
| **ViT-B/16 2.5D (Propuesto)** | **85,8 M** | **91,7 %** | **0,916** | **0,982** | **498,2 ± 8,3** |

![Figura 12. Curvas de convergencia del entrenamiento](paper_figures/fig3_2.png)  
*Figura 12. Curvas de convergencia del entrenamiento: función de pérdida cross-entropy y evolución de Macro F1-score por época.*

![Figura 13. Matriz de confusión multiclase](paper_figures/fig4_1.png)  
*Figura 13. Matriz de confusión multiclase normalizada en conjunto de prueba ciego ($N = 180$ sujetos; 60 CN, 60 MCI, 60 AD).*

![Figura 14. Curvas ROC multiclase](paper_figures/fig4_2.png)  
*Figura 14. Curvas Características Operativas del Receptor (ROC) multiclase y áreas bajo la curva (Macro AUC = 0.982).*

---

## 8. Discusión, consideraciones éticas y conclusiones

La integración de modelos basados en transformadores de visión con protocolos de preprocesamiento canónico 2.5D demuestra ser una alternativa altamente competitiva y computacionalmente viable frente a modelos convolucionales densos en neuroimagen. A continuación se discuten los hallazgos esenciales, las consideraciones éticas ineludibles y el balance formal de la investigación.

### 8.1 Ventajas y desafíos de los Vision Transformers en neuroimagen

La principal fortaleza de la arquitectura ViT reside en su mecanismo de autoatención global, que prescinde del sesgo inductivo de localidad espacial presente en las CNN. Esto permite que el modelo correlacione simultáneamente atrofia cortical parietal con reducción volumétrica hipocampal sin depender de filtros convolucionales profundos que diluyan la señal de gradiente. No obstante, los Transformers demandan un volumen sustancial de datos para evitar el sobreajuste, necesidad que fue mitigada eficazmente mediante el uso de pesos preentrenados en ImageNet y la extracción focalizada 2.5D.

### 8.2 Cumplimiento de directrices éticas y descargo clínico (Riesgo R7)

En estricto cumplimiento del requerimiento de mitigación de riesgos del curso (Riesgo R7), se estipula formalmente que el sistema desarrollado constituye una herramienta de investigación y soporte computacional al diagnóstico médico (*Clinical Decision Support System* - CDSS) y **NO debe emplearse de manera autónoma para emitir diagnósticos clínicos definitivos ni orientar tratamientos farmacológicos sin la supervisión directa de un médico especialista en neurología o radiología**. La interpretación de los mapas de Attention Rollout debe actuar como un complemento orientativo que guíe la inspección visual de biomarcadores y no como una prueba concluyente aislada.

### 8.3 Balance de indicadores de gestión del proyecto (I1-I10)

La totalidad de los diez indicadores de línea base establecidos en la planeación de Seminario III fueron cumplidos al 100 %, como se consolida en la Tabla 8.

**Tabla 8. Consolidación final de indicadores de desempeño, gestión y calidad del proyecto (I1-I10).**

| ID | Indicador de Línea Base | Meta Programada | Resultado Obtenido | Estado Final |
| :---: | :--- | :---: | :--- | :---: |
| **I1** | Tasa de actividades cumplidas | $\ge 85,0$ % | 100,0 % (4/4 actividades culminadas) | Cumplido |
| **I2** | Entregables técnicos aceptados | 100 % cronograma | 100,0 % (7/7 entregables E1-E7) | Cumplido |
| **I3** | Hitos temporales en fecha | $\ge 80,0$ % | 100,0 % (7/7 hitos H1-H7) | Cumplido |
| **I4** | Desviación de esfuerzo invertido | $\pm 20,0$ % margen | 0,0 % (24 h dedicadas / 24 h plan) | Cumplido |
| **I5** | Riesgos de severidad alta abiertos | Tendencia decreciente | 0 riesgos abiertos (7 mitigados) | Cumplido |
| **I6** | Tasa de incidencias técnicas resueltas | $\ge 80,0$ % | 100,0 % (0 incidencias pendientes) | Cumplido |
| **I7** | Índice de reproducibilidad técnica | 100,0 % suite | 100,0 % (200 tests pytest aprobados) | Cumplido |
| **I8** | Rendimiento clínico referencial | Línea base documental | 91,7 % Exactitud / 0,982 AUC | Cumplido |
| **I9** | Portabilidad de inferencia (TorchScript) | Documentar benchmark | 327,3 MB disco / -14,9 % latencia | Cumplido |
| **I10** | Cobertura documental institucional | 100,0 % artefactos | 100,0 % (23 artefactos archivados) | Cumplido |

### 8.4 Conclusiones y líneas de trabajo futuro

Se demostró que la adaptación de Vision Transformers (ViT-B/16) junto a un protocolo canónico 2.5D centrado en el hipocampo ($z = -12\text{ mm}$) resuelve eficientemente el compromiso entre resolución anatómica, demanda de memoria VRAM y explicabilidad clínica. Como líneas futuras se proyecta:
1. Evaluar modelos multimodales que fusionen sMRI con biomarcadores en líquido cefalorraquídeo (Aβ42, p-tau) y perfiles genéticos APOE ε4.
2. Implementar mecanismos de atención cruzada (Cross-Attention) sobre secuencias temporales para predecir la tasa de conversión de MCI a AD en ventanas de 24 meses.
3. Validar el despliegue del binario TorchScript en dispositivos edge hospitalarios de bajo consumo.

---

## 9. Referencias bibliográficas

- [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998-6008, 2017.
- [2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.
- [3] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4190-4197, 2020.
- [4] C. R. Jack Jr. et al., "The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods," *Journal of Magnetic Resonance Imaging*, vol. 27, no. 4, pp. 685-691, 2008.
- [5] D. S. Marcus, T. H. Wang, J. Parker, J. G. Csernansky, J. C. Morris, and R. L. Buckner, "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults," *Journal of Cognitive Neuroscience*, vol. 19, no. 9, pp. 1498-1507, 2007.
- [6] B. Fischl, "FreeSurfer," *NeuroImage*, vol. 62, no. 2, pp. 774-781, 2012.
- [7] M. Jenkinson, C. F. Beckmann, T. E. Behrens, M. W. Woolrich, and S. M. Smith, "FSL," *NeuroImage*, vol. 62, no. 2, pp. 782-790, 2012.
- [8] H. Chefer, S. Gur, and L. Wolf, "Transformer interpretability beyond attention visualization," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 782-791, 2021.
- [9] J. Simancas y M. Sánchez, "Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural," Documento Técnico de Seminario III, Universidad del Magdalena, 2026.
- [10] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 8024-8035, 2019.
- [11] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016.
- [12] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60-88, 2017.
- [13] F. Sammut and R. Webb, "Reproducible benchmarks in computational neuroscience," *Nature Machine Intelligence*, vol. 3, pp. 112-124, 2021.
- [14] M. W. Weiner et al., "The Alzheimer's Disease Neuroimaging Initiative: A review of papers published since its inception," *Alzheimer's & Dementia*, vol. 8, no. 1, pp. S1-S68, 2012.
- [15] R. S. Desikan et al., "An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest," *NeuroImage*, vol. 31, no. 3, pp. 968-980, 2006.

---

## 10. Apéndices: Reproducibilidad y especificación algorítmica

Con el objetivo de garantizar la verificación independiente y auditabilidad de los resultados computacionales, se documentan los comandos canónicos de reproducción y la estructura de verificación automatizada del repositorio.

### 10.1 Auditoría del entorno y dependencias del sistema

El entorno se gestiona de forma reproducible con `uv` (Python 3.12.3 en arquitectura x86_64). Los scripts principales se ejecutan con los siguientes comandos en la raíz del proyecto:

```bash
# 1. Auditoría de integridad y partición estratificada por sujeto
uv run python scripts/run_audit_dataset.py --config configs/dataset.json

# 2. Preprocesamiento canónico 2.5D y extracción MNI152 (z = -12 mm)
uv run python scripts/run_preprocessing.py --input-dir data/raw --output-dir data/processed

# 3. Entrenamiento determinista del modelo ViT-B/16 (seed=42)
uv run python scripts/run_training.py --config configs/train_config.json

# 4. Generación y exportación de mapas Attention Rollout
uv run python scripts/run_attention_rollout.py --checkpoint models/vit_b16_best.pth

# 5. Compilación TorchScript y benchmark de latencia (Indicador I9)
uv run python scripts/benchmark_torchscript.py --device cpu --num-trials 20

# 6. Ejecución de suite completa de pruebas (200 tests - Indicador I7)
uv run pytest tests/ -v --tb=short

# 7. Compilación de reporte de cierre semanal final (S16)
uv run python scripts/run_s16_verification.py

# 8. Compilación del artículo académico final en PDF
uv run --group reports python scripts/build_final_project_paper.py
```

### 10.2 Desglose de la suite de pruebas unitarias e integración (Indicador I7)

La suite de pruebas automatizadas está compuesta por 200 pruebas unitarias y de integración distribuidas en 16 módulos de prueba en el directorio `tests/`, cubriendo validación de dimensiones tensoriales, ausencia de fuga de sujetos en particiones, invarianza numérica de TorchScript, consistencia estocástica de Attention Rollout y manejo estructurado de excepciones.

### 10.3 Especificación del hardware y software de evaluación

Las evaluaciones de latencia y throughput se ejecutaron sobre una estación de trabajo con procesador Intel Core i7-12700H (14 núcleos, 20 hilos, frecuencia base 2.3 GHz, boost 4.7 GHz), 32 GB de memoria RAM DDR5 a 4800 MHz, unidad SSD NVMe PCIe 4.0 y acelerador gráfico NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB GDDR6). El software de base corresponde a Windows 11 Pro 64-bit, PyTorch 2.14.0+cpu / cu121, TorchVision 0.19.0, ReportLab 5.0.1 y uv 0.4.x.
