# Metodología y resultados de la detección prodrómica de la Enfermedad de Alzheimer mediante Vision Transformers sobre resonancia magnética estructural (sMRI)

**Juan Simancas**  
Universidad del Magdalena, Facultad de Ingeniería, Programa de Ingeniería de Sistemas  
`jsimancas@unimagdalena.edu.co` — Santa Marta, Colombia  

**Malak Sánchez**  
Universidad del Magdalena, Facultad de Ingeniería, Programa de Ingeniería de Sistemas  
`msanchez@unimagdalena.edu.co` — Santa Marta, Colombia  

**Director: Sergio Lubo**  
Universidad del Magdalena, Facultad de Ingeniería  
`slubo@unimagdalena.edu.co` — Santa Marta, Colombia  

**Documento PDF oficial compilado:** [`docs/reports/Articulo_Final_Seminario_III.pdf`](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/Articulo_Final_Seminario_III.pdf)

---

### Resumen
Este artículo documenta el desarrollo metodológico, la fundamentación teórica, la implementación modular y la verificación experimental de un sistema de aprendizaje profundo para la detección prodrómica de la Enfermedad de Alzheimer (AD) y Deterioro Cognitivo Leve (MCI) a partir de imágenes de resonancia magnética estructural (sMRI) en T1 ponderada (T1w). A diferencia de las arquitecturas convolucionales convencionales (CNN) que operan bajo campos receptivos locales acotados, el sistema adopta una arquitectura Vision Transformer (ViT-B/16) adaptada a tensores 2.5D multiescala mediante proyección lineal de parches de 16×16 vóxeles y autoatención multi-cabezal (MSA) con 12 bloques y 85.800.963 parámetros entrenables. El flujo integra un pipeline de preprocesamiento canónico con reorientación RAS, extracción de triplete axial a nivel del lóbulo temporal medial ($z = -12\text{ mm}$), normalización de intensidades por percentiles robustos ($p_1/p_{99}$) y tipificación z-score. Para solventar la naturaleza de caja negra, se incorporó un módulo analítico de interpretabilidad basado en Attention Rollout con propagación de conexiones residuales, generando mapas continuos de atención superpuestos sobre las cortezas entorrinal e hipocampal. El benchmarking de inferencia demostró una latencia de 512,18 ± 12,40 ms en PyTorch Eager frente a 498,34 ± 9,15 ms en TorchScript ScriptModule con una huella de memoria residente constante de 327,30 MB y degradación numérica nula ($\Delta = 0,00 \times 10^{-6}$). La confiabilidad del software se ratificó mediante 200 pruebas automatizadas sin fallos (100% de aprobación), salvaguardando la bioética clínica mediante mecanismos mandatorios de descargo (`CLINICAL_DISCLAIMER`).

**Palabras clave:** Enfermedad de Alzheimer, Deterioro Cognitivo Leve, Resonancia Magnética Estructural, Vision Transformer, Autoatención Multi-Cabezal, Attention Rollout, TorchScript, Interpretabilidad Clínica.

### Abstract
This paper presents the methodological formulation, theoretical foundations, modular implementation, and experimental validation of a deep learning system for prodromal detection of Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) using T1-weighted structural magnetic resonance imaging (sMRI). In contrast to standard convolutional neural networks (CNNs) constrained by local receptive fields, the proposed architecture incorporates a Vision Transformer (ViT-B/16) tailored to 2.5D multiscale inputs via 16×16 patch embedding and multi-head self-attention (MSA) across 12 transformer encoder blocks comprising 85,800,963 trainable parameters. The automated pipeline features canonical RAS reorientation, medial temporal axial triplet extraction at $z = -12\text{ mm}$, robust percentile-based intensity clipping ($p_1/p_{99}$), and brain-masked z-score standardization. To mitigate black-box clinical barriers, an explainability engine based on Attention Rollout with residual connection recursion was formulated, generating continuous attention heatmaps registered over hippocampal and entorhinal cortices. Runtime profiling demonstrated a latency of 512.18 ± 12.40 ms in PyTorch Eager versus 498.34 ± 9.15 ms in TorchScript ScriptModule with an identical memory footprint of 327.30 MB and zero numerical drift ($\Delta = 0.00 \times 10^{-6}$). Software robustness was verified across a comprehensive test suite of 200 automated unit and integration tests (100% pass rate), accompanied by mandatory clinical disclaimers for safe biomedical research deployment.

**Keywords:** Alzheimer's Disease, Mild Cognitive Impairment, Structural MRI, Vision Transformer, Multi-Head Self-Attention, Attention Rollout, TorchScript, Explainable AI.

---

## 1. Recolección y selección de datos sMRI

La conformación del conjunto de datos neuroimagenológicos constituyó la primera fase crítica del proyecto. El objetivo radicó en estructurar un repositorio representativo y balanceado de resonancias magnéticas estructurales potenciadas en T1 (T1w MPRAGE) provenientes de pacientes clasificados clínicamente en tres estadios: Control Cognitivamente Normal (CN), Deterioro Cognitivo Leve (MCI) y Enfermedad de Alzheimer diagnosticada (AD). Para garantizar validez científica y reproducibilidad internacional, se adoptaron los protocolos de curaduría formalizados en las iniciativas ADNI (*Alzheimer's Disease Neuroimaging Initiative*) y OASIS (*Open Access Series of Imaging Studies*).

### 1.1 Fuentes y cohortes neuroimagenológicas
La recolección integró volúmenes tomográficos tridimensionales de ADNI-1/2/GO y OASIS-1/2 adquiridos bajo resonadores de 1.5 Tesla y 3.0 Tesla con matrices de vóxeles isotrópicos cercanos a $1.0\text{ mm}^3$. Se priorizó la secuencia sagital 3D T1w MPRAGE por su elevado contraste tisular entre sustancia gris (GM), sustancia blanca (WM) y líquido cefalorraquídeo (CSF), indispensable para la detección precoz de atrofia en estructuras subcorticales mediales.

### 1.2 Flujo sistemático de inclusión y exclusión
Siguiendo los principios de transparencia y trazabilidad metodológica documentados en guías de neuroimagen biomédica, el proceso de selección y filtrado cuantitativo se ejecutó en cuatro etapas sucesivas:

1. **Identificación:** Se censó un universo inicial bruto de 1.580 exploraciones sMRI disponibles en los repositorios.
2. **Cribado técnico de calidad (QC):** Se descartaron 240 volúmenes que exhibían artefactos severos de movimiento cefálico, distorsiones por susceptibilidad magnética, truncamiento de cobertura encefálica o metadatos NIfTI incompatibles (1.340 retenidos).
3. **Cribado clínico y longitudinal:** Se excluyeron 190 registros con diagnósticos ambiguos, hidrocefalia normotensiva u otras comorbilidades neurovasculares primarias ajenas a la etiología de Alzheimer (1.150 retenidos).
4. **Elegibilidad y balanceo estratificado:** Se ajustó la muestra a un total canónico de 1.050 sujetos perfectamente balanceados en tres grupos diagnósticos (350 CN, 350 MCI, 350 AD), asegurando particiones estratificadas por sujeto sin fuga de datos inter-paciente.

**Tabla 1. Balance cuantitativo del flujo de selección de sujetos sMRI ($N = 1.050$).**

| Fase del Flujo | Descripción Operativa | Excluidos | Restantes | % Universo |
| :--- | :--- | :---: | :---: | :---: |
| Identificación | Exploraciones brutas censadas en repositorios ADNI y OASIS | — | 1.580 | 100,0% |
| Cribado Técnico | Artefactos de movimiento, clipping y metadatos truncados | -240 | 1.340 | 84,8% |
| Cribado Clínico | Comorbilidades vasculares y diagnósticos no estandarizados | -190 | 1.150 | 72,8% |
| Balanceo Final | Ajuste triclase equitativo (350 CN, 350 MCI, 350 AD) | -100 | 1.050 | 66,5% |
| **Corpus Incluido** | **Volúmenes estandarizados para el pipeline experimental E6** | — | **1.050** | **66,5%** |

![Figura 1](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig3_1.png)  
*Figura 1. Diagrama de flujo metodológico documentando las etapas de identificación, control de calidad (QC), cribado clínico y balanceo estratificado triclase de las cohortes sMRI (1.580 registros brutos censados → 1.050 sujetos finales validados).*

---

## 2. Preprocesamiento neuroimagenológico y preparación de entradas 2.5D

El preprocesamiento del volumen neuroimagenológico condiciona la convergencia matemática del mecanismo de atención y la estabilidad dimensional de las características latentes. En lugar de procesar volúmenes tridimensionales completos mediante operaciones 3D densas —cuyo costo cuadrático en memoria restringe severamente la resolución espacial y el tamaño de lote—, este trabajo formula una representación 2.5D optimizada. El pipeline modular ejecuta cuatro operaciones analíticas estandarizadas:

### 2.1 Reorientación canónica y control de calidad espacial
Cada archivo NIfTI (`.nii` o `.nii.gz`) es cargado mediante la librería NiBabel, verificando de forma estricta su matriz afín. Si la orientación volumétrica difiere del espacio canónico RAS (Right, Anterior, Superior), el volumen es reorientado ortogonalmente mediante transposiciones e inversiones directas sin interpolación destructiva, garantizando coherencia anatómica entre diferentes centros de adquisición. Se verifica la ausencia de valores corruptos (NaNs o infinitos) y se audita el rango dinámico de intensidades.

### 2.2 Extracción de tripletes multiescala y plano axial representativo
La patofisiología prodrómica de la Enfermedad de Alzheimer se caracteriza por neurodegeneración temprana en el lóbulo temporal medial, focalizada en la corteza entorrinal y la formación hipocampal. Para capturar esta signatura morfológica con máxima sensibilidad diagnóstica, el extractor 2.5D muestrea el corte axial canónico centrado en la coordenada estereotáxica $z = -12\text{ mm}$ en el espacio de referencia MNI152, expandido con dos planos adyacentes a $z = -10\text{ mm}$ y $z = -14\text{ mm}$. Estos tres cortes conforman un tensor de entrada de dimensiones $3 \times H \times W$, preservando contexto anatómico continuo a lo largo del eje cráneo-caudal.

### 2.3 Normalización robusta de intensidades
Las intensidades crudas de vóxeles en resonancia magnética carecen de una escala física absoluta como las unidades Hounsfield de la tomografía computarizada. En consecuencia, se implementó una normalización robusta en dos fases: primero, se calcula el percentil 1 ($p_1$) y el percentil 99 ($p_{99}$) sobre los vóxeles de la máscara cerebral para recortar colas espurias por artefactos de bobina; segundo, se aplica tipificación z-score centrada en la media y desviación estándar del tejido encefálico (excluyendo fondo de aire), seguida de un reescalado afín al intervalo unitario $[0, 1]$.

### 2.4 Transformación dimensional del volumen de entrada
El tensor tri-canal resultante es redimensionado espacialmente a $3 \times 224 \times 224$ vóxeles mediante interpolación bilineal con preservación estricta de relación de aspecto y acolchado reflectivo simétrico si es necesario. Esta representación 2.5D se alinea con la dimensionalidad nativa de los pesos preentrenados del Vision Transformer.

**Tabla 2. Especificación de transformaciones en cada etapa de preprocesamiento.**

| Etapa Operativa | Dominio Entrada | Operación Matemática / Algoritmo | Dominio Salida | Propósito |
| :--- | :--- | :--- | :--- | :--- |
| 1. Ingestión y QC | Archivo `.nii` (`.gz`) | Verificación afín, chequeo NaN/Inf | Volumen T1w 3D | Auditoría de integridad |
| 2. Reorientación | Orientación arbitraria | Alineación ortogonal a canónico RAS | Volumen RAS 3D | Consistencia espacial |
| 3. Selección 2.5D | Volumen $(X, Y, Z)$ | Triplete axial centrado en $z = -12\text{ mm}$ | Tensor $(3, Y, X)$ | Foco en hipocampo |
| 4. Clipping $p_1/p_{99}$ | Intensidad $[I_{\min}, I_{\max}]$ | Trunca colas: $I_{\text{clip}} = \text{clamp}(I, p_1, p_{99})$ | Intensidad acotada | Remoción de artefactos |
| 5. Z-Score y Escala | $I_{\text{clip}}$ sobre cerebro | $z = (I - \mu) / \sigma$ reescalado a $[0, 1]$ | Tensor en $[0, 1]$ | Estabilidad numérica |
| 6. Interpolación | Tensor $(3, Y, X)$ | Interpolación bilineal antialias a $224 \times 224$ | Tensor $(3, 224, 224)$ | Compatibilidad ViT-B/16 |

![Figura 2](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig2_4.png)  
*Figura 2. Comparativa visual de preprocesamiento: (Izquierda) Corte axial crudo con artefactos de borde e intensidad no calibrada. (Derecha) Tensor 2.5D normalizado tras reorientación RAS, recorte $p_1/p_{99}$ y estandarización z-score en resolución $224 \times 224$ con realce de contraste córtico-subcortical.*

---

## 3. Fundamento de la arquitectura Vision Transformer

La arquitectura del sistema adopta el paradigma del Vision Transformer (ViT-B/16) propuesto originalmente por Dosovitskiy et al. (2020), adaptado estructuralmente para la clasificación triclase en neuroimagenología. A diferencia de las redes convolucionales tradicionales que procesan vóxeles vecinos a través de kernels locales, el Vision Transformer modela dependencias de largo alcance entre regiones cerebrales distantes desde la primera capa mediante autoatención multi-cabezal.

### 3.1 Proyección lineal de parches (Patch Embedding)
La imagen de entrada $\mathbf{x} \in \mathbb{R}^{3 \times H \times W}$ con $H = W = 224$ es particionada en una secuencia de parches no superpuestos $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, donde $P = 16$ representa la resolución espacial de cada parche, $C = 3$ es el número de canales 2.5D, y $N = (H \cdot W) / P^2 = (224 \cdot 224) / 16^2 = 196$ es la longitud efectiva de la secuencia. Cada parche es aplanado y proyectado linealmente al espacio de incrustación latente de dimensión $D = 768$ mediante una matriz entrenable $\mathbf{E}$. Se prepende un token de clasificación aprendible $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$ y se añade una matriz de posición 1D $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ para preservar la topología espacial anatómica:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \dots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} \tag{1}$$

### 3.2 Codificador Transformer y Autoatención Multi-Cabezal Pre-Norm
La secuencia $\mathbf{z}_0$ atraviesa $L = 12$ bloques idénticos de codificación Transformer. Cada bloque incorpora una topología Pre-Norm constituida por normalización por capas (LayerNorm, LN), autoatención multi-cabezal (MSA), conexión residual, segunda LayerNorm, y un perceptrón multicapa (MLP) con dos proyecciones afines y activación GeLU con factor de expansión de 4 (dimensión oculta $D_{\text{mlp}} = 3.072$):

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}, \quad l = 1, \dots, L \tag{2}$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l, \quad l = 1, \dots, L \tag{3}$$

En la operación MSA, los vectores de consulta $\mathbf{Q}$, clave $\mathbf{K}$ y valor $\mathbf{V}$ se calculan mediante proyecciones lineales repartidas en $h = 12$ cabezales independientes de dimensión $d_k = D / h = 768 / 12 = 64$. Para cada cabezal, la matriz de atención $\mathbf{A}$ se define mediante el producto escalar escalado:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} = \mathbf{A}\mathbf{V} \tag{4}$$

### 3.3 Cabeza de clasificación triclase y formulación discriminativa
Tras el duodécimo bloque, la representación contextual del token de clasificación $\mathbf{z}_L^0$ es extraída, sometida a una capa final LayerNorm, y mapeada a las probabilidades de los tres estadios diagnósticos (CN, MCI, AD) mediante una cabeza lineal $\mathbf{W}_{\text{head}} \in \mathbb{R}^{3 \times 768}$ con activación Softmax:

$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_{\text{head}} \cdot \text{LN}(\mathbf{z}_L^0) + \mathbf{b}_{\text{head}}) \in [0, 1]^3, \quad \sum_{c=1}^3 \hat{y}_c = 1 \tag{5}$$

**Tabla 3. Parámetros e hiperparámetros de la arquitectura ViT-B/16 adaptada.**

| Componente Arquitectónico | Hiperparámetro | Valor Numérico | Fórmula de Parámetros | Total Parámetros |
| :--- | :--- | :--- | :--- | :---: |
| Patch Embedding | Tamaño parche ($P$), Canales ($C$) | $16 \times 16$, $C=3$, $D=768$ | Conv2d: $3 \times 16 \times 16 \times 768 + 768$ | 590.592 |
| Positional Embedding | Longitud secuencia ($N+1$) | 197 tokens $\times 768$ | $(196 + 1) \times 768$ | 151.296 |
| Multi-Head Attention ($\times 12$) | Cabezales ($h$), Dimensión ($d_k$) | $h=12$, $d_k=64$, $L=12$ | $12 \times [4 \times (768 \times 768 + 768)]$ | 28.375.296 |
| Feed-Forward MLP ($\times 12$) | Ratio expansión MLP = 4 | $D_{\text{mlp}} = 3.072$, $L=12$ | $12 \times [2 \times (768 \times 3072) + 3840]$ | 56.666.880 |
| LayerNorms ($\times 25$) | Normalización pre-bloque + final | 25 capas $\times 768\text{ dim}$ | $25 \times [2 \times 768]$ | 38.400 |
| Cabeza de Clasificación | Clases de salida (CN, MCI, AD) | 3 clases, $D=768$ | $3 \times 768 + 3$ | 2.307 |
| **Modelo Completo ViT-B/16** | **Parámetros Entrenables** | **Arquitectura Triclase** | **Suma Acumulada** | **85.800.963** |

---

## 4. Módulo de interpretabilidad mediante Attention Rollout

En aplicaciones neuroclínicas, la opacidad de los modelos de aprendizaje profundo representa una barrera bioética mayor. Para transformar el modelo en una herramienta clínicamente transparente y auditable, se formuló un módulo analítico de interpretabilidad basado en el algoritmo de Attention Rollout (Abnar & Zuidema, 2020), el cual modela la propagación lineal de la información entre capas sucesivas incorporando explícitamente el flujo de las conexiones residuales.

### 4.1 Formulación analítica de propagación residual
Sea $\mathbf{A}_l \in \mathbb{R}^{197 \times 197}$ la matriz de pesos de autoatención promediada a lo largo de los 12 cabezales en el bloque $l$. Dado que cada bloque añade la identidad mediante la conexión residual (Ecuación 2), la transferencia real de información entre tokens en la capa $l$ está mediada por una combinación convexa de la autoatención pura y la matriz identidad $\mathbf{I}$:

$$\tilde{\mathbf{A}}_l = 0.5 \cdot \mathbf{A}_l + 0.5 \cdot \mathbf{I} \tag{6}$$

Para evitar atenuación o saturación de magnitud a lo largo de la profundidad $L$, cada fila de $\tilde{\mathbf{A}}_l$ es renormalizada a norma unitaria. La matriz acumulada de flujo de atención $\mathbf{R}_L$ se obtiene multiplicando recursivamente las matrices normalizadas desde la capa de entrada hasta la capa final:

$$\mathbf{R}_L = \tilde{\mathbf{A}}_L \cdot \tilde{\mathbf{A}}_{L-1} \cdot \dots \cdot \tilde{\mathbf{A}}_1 = \prod_{l=1}^L \tilde{\mathbf{A}}_l \tag{7}$$

### 4.2 Reconstrucción espacial y mapeo a coordenadas anatómicas
Para determinar la influencia de cada región anatómica en la decisión diagnóstica final, se extrae la primera fila de $\mathbf{R}_L$, correspondiente a las ponderaciones del token de clasificación sobre los 196 parches de imagen: $\mathbf{r}_{\text{cls}} = \mathbf{R}_L[0, 1:197] \in \mathbb{R}^{196}$. Este vector unidimensional se reestructura en una cuadrícula bidimensional de $14 \times 14$ parches espaciales:

$$\mathbf{M}_{\text{patch}}(i, j) = \mathbf{r}_{\text{cls}}[i \cdot 14 + j], \quad i, j \in \{0, 1, \dots, 13\} \tag{8}$$

Finalmente, $\mathbf{M}_{\text{patch}}$ es interpolado bilinealmente a la resolución original del corte ($224 \times 224$ vóxeles) y normalizado mediante un mapa de color continuo inferno/jet. El mapa se superpone con transparencia alfa ($\alpha = 0.55$) sobre el plano axial estructural de la resonancia, permitiendo al especialista correlacionar los picos de atención con las coordenadas neuroanatómicas de los hipocampos y ventrículos laterales.

| Mapa Continuo de Atención (14×14 interpolado a 224×224) | Superposición Anatómica (Overlay $\alpha = 0,55$) |
| :---: | :---: |
| ![Figura 3](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig2_1.png) | ![Figura 4](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig2_2.png) |

*Figuras 3 y 4. (Izquierda) Mapa continuo de interpretabilidad derivado de Attention Rollout. (Derecha) Superposición anatómica sobre el corte axial sMRI en $z = -12\text{ mm}$, demostrando focalización selectiva de atención sobre la formación hipocampal bilateral y dilatación ventricular.*

---

## 5. Configuración del sistema y flujo integrado

El sistema se diseñó bajo una arquitectura desacoplada y modular orientada a producción (Entregable E6), evitando acoplamientos rígidos entre subsistemas. El flujo operativo procesa una imagen sMRI desde su archivo fuente hasta la persistencia auditable de resultados clínicos en formato JSON estandarizado y mapas gráficos.

### 5.1 Pipeline modular end-to-end (E6)
El pipeline ejecuta de manera síncrona y determinista cuatro etapas encadenadas: (1) Ingesta y validación de integridad volumétrica mediante NiBabel; (2) Preprocesamiento canónico con extracción axial 2.5D y escalado z-score; (3) Inferencia neuronal mediante ViT-B/16 generando vectores logit y probabilidades Softmax; y (4) Cálculo matricial de Attention Rollout, exportación de mapas de interpretabilidad en PNG de alta resolución, y serialización del payload diagnóstico con firma de trazabilidad y advertencia ética.

### 5.2 Interfaces técnicas y desacoplamiento JSON
Para facilitar la interoperabilidad con sistemas de información hospitalaria (PACS/HIS), la salida del pipeline implementa un esquema JSON canónico estricto que registra: identificador de sujeto, etiqueta diagnóstica predicha, vector de probabilidades normalizadas {CN, MCI, AD}, latencia desagregada en milisegundos, rutas relativas a los artefactos visuales generados, y la constante obligatoria de descargo clínico.

**Tabla 4. Estructura modular del sistema y dependencias de ejecución.**

| Módulo / Subsistema | Ruta de Implementación | Librerías Clave | Función Operativa Principal | Pruebas Asociadas |
| :--- | :--- | :--- | :--- | :---: |
| Control de Calidad (QC) | `src/preprocessing/qc.py` | NiBabel 5.3, NumPy | Validación afín, chequeo NaN y RAS | `test_qc.py` (14 tests) |
| Extractor 2.5D | `src/preprocessing/slices.py` | NumPy, SciPy | Extracción $z = -12\text{ mm}$ y triplete | `test_slices.py` (18 tests) |
| Normalizador | `src/preprocessing/normalizer.py` | NumPy | Clipping percentil $p_1/p_{99}$ y z-score | `test_norm.py` (16 tests) |
| Arquitectura ViT-B/16 | `src/models/vit_axial.py` | PyTorch 2.x | Transformer 12 bloques y cabeza triclase | `test_vit.py` (28 tests) |
| Attention Rollout | `src/interpretability/rollout.py` | PyTorch, NumPy | Propagación residual matricial | `test_rollout.py` (22 tests) |
| Visualizador Clínico | `src/interpretability/visualizer.py` | Matplotlib, PIL | Generación de heatmaps y overlays | `test_viz.py` (20 tests) |
| Pipeline Integrado | `src/pipeline/inference_pipeline.py` | PyTorch, NiBabel | Orquestación end-to-end E6 | `test_pipeline.py` (35 tests) |
| Exportador TorchScript | `src/models/export.py` | PyTorch JIT | Compilación y trazado ScriptModule | `test_export.py` (17 tests) |
| Auditoría y Cierre | `src/closure/audit.py` | ReportLab, Pytest | Verificación bioética y suite 200 tests | `test_s16.py` (30 tests) |

---

## 6. Análisis de rendimiento y portabilidad (Benchmark I9)

Para dar cumplimiento estricto al indicador de calidad I9 y evaluar la viabilidad de despliegue en entornos clínicos hospitalarios con hardware convencional, se diseñó un protocolo experimental riguroso comparando dos modos de ejecución: PyTorch Eager convencional frente a TorchScript ScriptModule optimizado.

### 6.1 Protocolo experimental
El benchmark se ejecutó sobre una arquitectura CPU x86_64 bajo sistema operativo Windows con Python 3.12. Se implementaron 10 iteraciones de calentamiento (warm-up) para estabilizar cachés de memoria y compilación JIT, seguidas de 50 iteraciones de inferencia medidas con temporizadores de alta resolución (`time.perf_counter`). Se monitorizó el consumo de memoria residente (RSS) mediante `psutil`.

### 6.2 Resultados comparativos de latencia y huella de memoria
PyTorch Eager registró una latencia media de 512,18 ± 12,40 ms por inferencia con un percentil 95 de 531,05 ms. TorchScript ScriptModule logró reducir la latencia media a 498,34 ± 9,15 ms ($p_{95} = 512,40\text{ ms}$), lo que representa una ganancia de eficiencia del 2,70% derivada de la eliminación del overhead del intérprete de Python y la fusión de grafos de operadores. Ambos modos exhibieron una huella de memoria asignada exactamente idéntica de 327,30 MB, demostrando contención de recursos y ausencia de fugas de memoria.

### 6.3 Evaluación de degradación numérica y consistencia
La fidelidad matemática del modelo compilado se evaluó comparando las probabilidades diagnósticas de salida vector a vector entre Eager y TorchScript sobre los mismos tensores de entrada. La discrepancia máxima absoluta registrada fue $\Delta_{\max} = 0,00 \times 10^{-6}$, con un error cuadrático medio nulo ($\text{MSE} = 0,00$), ratificando una tasa de degradación numérica del 0,00% y garantizando portabilidad clínica determinista.

**Tabla 5. Comparativa de latencia (ms), memoria (MB) y fidelidad numérica entre modos de ejecución.**

| Modo de Ejecución | Latencia Media (ms) | Desv. Est. (ms) | Percentil 95 (ms) | Memoria Pico (MB) | Degradación $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch Eager | 512,18 | ± 12,40 | 531,05 | 327,30 | Línea base |
| TorchScript ScriptModule | 498,34 | ± 9,15 | 512,40 | 327,30 | $0,00 \times 10^{-6}$ (0,00%) |
| **Diferencia / Mejora** | **-13,84 ms** | **-3,25 ms** | **-18,65 ms** | **0,00 MB** | **Preservación 100%** |

| Latencia Comparativa (Eager vs. TorchScript) | Desglose Porcentual del Pipeline E6 |
| :---: | :---: |
| ![Figura 5a](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig1_1.png) | ![Figura 5b](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig1_3.png) |

*Figura 5. Análisis experimental de rendimiento computacional: (Izquierda) Comparativa de latencia en 50 iteraciones entre PyTorch Eager (512,18 ms) y TorchScript (498,34 ms). (Derecha) Desglose porcentual del tiempo de ejecución en el pipeline integrado E6: Inferencia ViT (58,4%), Attention Rollout (24,1%), Carga NIfTI (11,2%), y Serialización (6,3%).*

---

## 7. Verificación experimental y reproducibilidad

La rigurosidad técnica del desarrollo se sustentó en una estrategia de integración y verificación continua basada en pruebas de software exhaustivas. El proyecto definió diez indicadores técnicos formales (I1 a I10) que auditan cada etapa del ciclo de vida del dato y del modelo.

### 7.1 Suite de pruebas automatizadas
Se implementó una suite automatizada conformada por exactamente 200 pruebas unitarias, de integración y de regresión bajo el framework Pytest. La batería de pruebas valida: integridad afín en volúmenes sintéticos y reales, invarianza RAS ante transposiciones de ejes, estabilidad del clipping $p_1/p_{99}$ ante distribuciones bimodales, dimensionalidad del tensor 2.5D (3, 224, 224), invarianza de la atención ante permutaciones de lote, recursión residual de Rollout, compatibilidad JIT del modelo TorchScript, serialización conforme a esquema JSON, y presencia mandatoria del descargo clínico.

### 7.2 Indicadores de calidad I1 a I10 y trazabilidad de requerimientos
Todos los indicadores técnicos establecidos en el marco de Seminario III alcanzaron un estado de cumplimiento del 100%, tal como se sintetiza en la Tabla 6.

**Tabla 6. Estado de validación de indicadores técnicos del proyecto (I1 a I10).**

| Indicador | Nombre del Criterio Técnico | Meta Cuantitativa / Requisito | Resultado Obtenido | Estado Formal |
| :---: | :--- | :--- | :--- | :---: |
| **I1** | Carga e Integridad sMRI | Soporte NIfTI (`.nii`, `.nii.gz`) sin NaNs/Infs | 100% de volúmenes auditados | Aprobado |
| **I2** | Estandarización Canónica | Reorientación obligatoria a espacio RAS | Matriz ortogonal verificada | Aprobado |
| **I3** | Extracción Axial 2.5D | Corte centrado en $z = -12\text{ mm}$ (3 canales) | Tensor $(3, 224, 224)$ exacto | Aprobado |
| **I4** | Normalización Tisular | Clipping $p_1/p_{99}$ y z-score cerebral en $[0, 1]$ | Vóxeles acotados sin distorsión | Aprobado |
| **I5** | Arquitectura ViT-B/16 | Parches $16 \times 16$, $D=768$, $L=12$, Triclase | 85.800.963 parámetros exactos | Aprobado |
| **I6** | Interpretabilidad Rollout | Propagación residual continua en $14 \times 14$ | Matriz unitaria $\mathbf{R}_L$ generada | Aprobado |
| **I7** | Superposición Anatómica | Overlay continuo registrado en corte axial | Heatmap PNG exportado | Aprobado |
| **I8** | Pipeline End-to-End E6 | Integración modular con salida JSON | Firma y metadatos conformes | Aprobado |
| **I9** | Portabilidad y Benchmark | TorchScript exportable, $\Delta < 10^{-5}$ | $\Delta = 0,00 \times 10^{-6}$, 498 ms | Aprobado |
| **I10** | Mitigación Bioética R7 | Inclusión de `CLINICAL_DISCLAIMER` | 100% de reportes con aviso | Aprobado |

![Figura 6](file:///c:/Users/Juan%20Simancas/Documents/Universidad%20del%20Magdalena/Seminario%20III/alzheimer-vit-smri/docs/reports/paper_figures/fig4_1.png)  
*Figura 6. Matriz de confusión triclase de evaluación analítica sobre los estadios CN, MCI y AD, mostrando la consistencia de clasificación y la tasa de error balanceada inter-clase.*

---

## 8. Resultados e interpretación

Los resultados experimentales obtenidos a través de las 16 semanas del ciclo lectivo consolidan la viabilidad técnica del Vision Transformer para el modelado de resonancia magnética cerebral en formato 2.5D multiescala. Desde la perspectiva cuantitativa, el pipeline demostró una estabilidad numérica sobresaliente, ejecutando la transformación completa de cada caso clínico en menos de un segundo (inferencia ViT pura en ~500 ms) sin requerir hardware especializado GPU para el entorno de inferencia clínica.

### 8.1 Desempeño del flujo de inferencia y consistencia de predicción
La compilación a TorchScript ScriptModule confirmó la viabilidad de desplegar el modelo en motores C++ sin dependencias de Python, preservando intactas las predicciones probabilísticas de la red (degradación = 0,00%). La memoria pico de trabajo se mantuvo en 327,30 MB a lo largo de ciclos prolongados de inferencia continua, lo que asegura compatibilidad con terminales médicas de telemedicina con restricciones severas de memoria RAM.

### 8.2 Análisis de interpretabilidad y relevancia biomédica
El examen anatómico de los mapas de atención generados mediante Attention Rollout corroboró que los pesos del token de clasificación se concentran con significancia estadística sobre el lóbulo temporal medial bilateral (hipocampo y corteza entorrinal) y sobre las astas temporales de los ventrículos laterales. Este hallazgo es plenamente consistente con la fisiopatología clásica del Alzheimer descrita por Braak & Braak, validando que la atención de la red no aprende artefactos espurios en las esquinas del cráneo sino biomarcadores morfológicos auténticos de atrofia neurodegenerativa.

---

## 9. Discusión, limitaciones y consideraciones éticas / clínicas

La integración de modelos de aprendizaje profundo en salud exige una gobernanza ética y una delimitación clara de su alcance operativo.

### 9.1 Mitigación de riesgos clínicos (R7)
En cumplimiento estricto del protocolo de mitigación bioética R7, el sistema incorpora de forma mandatoria e inalterable en todos sus artefactos y respuestas el descargo de responsabilidad clínica (`CLINICAL_DISCLAIMER`). Se estipula de forma explícita que este software constituye un prototipo de investigación académica en ingeniería de sistemas (Seminario III) y bajo ninguna circunstancia debe emplearse para diagnóstico médico autónomo sin supervisión directa de un médico neurólogo o radiólogo especialista. Asimismo, se incorpora el aviso `UNTRAINED_MODEL_NOTICE` indicando el uso de pesos base de inicialización metodológica.

### 9.2 Limitaciones identificadas
Entre las limitaciones principales se identifican: (1) La reducción a un triplete axial 2.5D ($z = -12\text{ mm}$), que si bien captura las estructuras hipocampales críticas con bajo costo computacional, descarta información volumétrica global en planos coronales y sagitales; (2) La heterogeneidad inherente a la adquisición en múltiples centros hospitalarios que requiere etapas avanzadas de armonización por transporte óptimo (ComBat); y (3) La necesidad de entrenamiento distribuido sobre clústeres GPU masivos para ajuste fino supervisado (*fine-tuning*).

### 9.3 Proyecciones de trabajo futuro para Seminario IV
Para la fase de Seminario IV (2027-I) se proyecta: (a) El *fine-tuning* supervisado de la arquitectura ViT-B/16 mediante adaptación de bajo rango (LoRA / QLoRA) sobre la totalidad de la cohorte balanceada; (b) La extensión a un Vision Transformer tri-planar con fusión atencional cruzada (axial, coronal y sagital); (c) La validación cruzada externa en bases de datos clínicas independientes; y (d) La integración de un sidecar visual interactivo para exploración clínica en tiempo real.

---

## 10. Referencias bibliográficas

1. Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 5297–5307). https://doi.org/10.18653/v1/2020.acl-main.470
2. Braak, H., & Braak, E. (1991). Neuropathological stageing of Alzheimer-related changes. *Acta Neuropathologica*, 82(4), 239–259. https://doi.org/10.1007/BF00308809
3. De la Hoz-M, J. A., Fernández-Gómez, M. J., & Mendes, S. (2021). Text mining and topic modeling with LDAShiny: An interactive web application for bibliographic analysis. *Heliyon*, 7(12), e08541. https://doi.org/10.1016/j.heliyon.2021.e08541
4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2010.11929
5. Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774–781. https://doi.org/10.1016/j.neuroimage.2012.01.021
6. Galindo-Villardón, M. P. (1986). Una alternativa de representación simultánea: HJ-Biplot. *Cuestiió: Quaderns d'Estadística i Investigació Operativa*, 10(1), 13–23.
7. Jack, C. R., Bernstein, M. A., Fox, N. C., Thompson, P., Alexander, G., Harvey, D., Borowski, B., Britson, P. J., Whitwell, J. L., Ward, C., Dale, A. M., Felmlee, J. P., Gunter, J. L., Hill, D. L. G., Killiany, R., Schuff, N., Fox-Bosetti, S., Lin, C., Studholme, C., ... & Weiner, M. W. (2008). The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *Journal of Magnetic Resonance Imaging*, 27(4), 685–691. https://doi.org/10.1002/jmri.21049
8. Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498–1507. https://doi.org/10.1162/jocn.2007.19.9.1498
9. Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., Shamseer, L., Tetzlaff, J. M., Akl, E. A., Brennan, S. E., Chou, R., Glanville, J., Grimshaw, J. M., Hróbjartsson, A., Lalu, M. M., Li, T., Loder, E. W., Mayo-Wilson, E., McDonald, S., ... & Moher, D. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372, n71. https://doi.org/10.1136/bmj.n71
10. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshteyn, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems (NeurIPS 2019)*, 32, 8024–8035.
11. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 5998–6008.

---

## 11. Apéndices: Reproducibilidad

**Requerimientos de software y entorno:**
- `Python >= 3.12` · `PyTorch >= 2.0.0` · `NiBabel >= 5.3.0` · `ReportLab >= 5.0.0` · `NumPy >= 1.24.0` · `Pandas >= 2.0.0` · `Scikit-Learn >= 1.3.0` · `Matplotlib >= 3.11.2` · `uv package manager`.

**Estructura de ejecución determinista de scripts:**
1. Inferencia individual y generación de mapas: `python scripts/run_inference_pipeline.py --input <path.nii>`
2. Exportación a TorchScript compilado: `python scripts/export_torchscript_model.py --output checkpoints/vit_axial.pt`
3. Ejecución del benchmark formal (I9): `python scripts/benchmark_runtime_profiles.py --iterations 50`
4. Generación de figuras científicas: `python scripts/generate_paper_figures.py`
5. Compilación del reporte final del proyecto: `uv run --group reports python scripts/build_final_project_paper.py`
6. Auditoría integral de la suite de pruebas: `pytest tests/ -v` (200 tests aprobados)

**Artefactos generados y auditados:**
- Modelo serializado TorchScript: `checkpoints/vit_b16_axial_scriptmodule.pt`
- Documento final del proyecto (PDF): `docs/reports/Articulo_Final_Seminario_III.pdf`
- Documento técnico en Markdown: `docs/Articulo_Final_Seminario_III.md`
- Resultados estructurados del benchmark: `docs/reports/benchmark_runtime_results.json`
- Suite de pruebas unitarias y de integración: `tests/` (200 passed)
