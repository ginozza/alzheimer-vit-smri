# Implementación S9 de arquitectura ViT y entrenamiento futuro

## 1. Propósito y relación con la línea base

La Semana 9 corresponde a la implementación de la arquitectura ViT-B/16 triclase, la definición de la salida del modelo y la validación estructural de dimensiones. Esta actividad aporta al Entregable E4, Código fuente del sistema, cuya aceptación formal está programada para la Semana 11. Por tanto, este documento registra avance técnico de E4 y no declara el cierre anticipado del entregable.

La arquitectura quedó implementada durante el cierre de E3 en la Semana 8. En S9 se reforzó su contrato de entrada, se volvió a validar la salida triclase y se inició el módulo de entrenamiento futuro previsto en el seguimiento de S8. No se realizó entrenamiento, ajuste de hiperparámetros ni evaluación diagnóstica.

## 2. Alcance implementado

### 2.1 Arquitectura y salida triclase

El componente `VisionTransformer` conserva la configuración ViT-B/16 definida en E3:

- Entrada: tensor `float32` con forma `(B, 3, 224, 224)`.
- Embeddings: 196 parches de 16 x 16 más el token de clasificación.
- Codificador: 12 bloques Transformer, dimensión 768 y 12 cabezas de atención.
- Salida: logits con forma `(B, 3)` en el orden `CN`, `MCI`, `AD`.
- Parámetros entrenables: 85.800.963.

`PatchEmbedding` ahora rechaza de forma explícita tensores con rango, canales, dimensiones espaciales o tipo incompatibles. Esto transforma errores internos poco claros en validaciones estables del contrato E2-E4.

### 2.2 Configuración de entrenamiento futuro

`TrainingConfig` externaliza los parámetros necesarios para una ejecución posterior:

- Optimizador AdamW.
- Tasa de aprendizaje, decaimiento de pesos, betas y epsilon.
- Suavizado de etiquetas opcional.
- Recorte de norma de gradiente.
- Pesos de clase opcionales para CN, MCI y AD.

La configuración está disponible en `configs/e4_training.json`. El valor de `class_weights` permanece nulo porque el repositorio no contiene los manifiestos completos de la partición de entrenamiento real. Los pesos deben calcularse exclusivamente con las etiquetas de entrenamiento para evitar fuga de información desde validación o prueba.

### 2.3 Pérdida ponderada y optimizador

El módulo `src/training/losses.py` implementa pesos balanceados por frecuencia inversa:

`peso_clase = total_muestras / (numero_clases x muestras_clase)`

También crea la función de entropía cruzada ponderada. Se rechazan distribuciones sin representación de alguna clase, etiquetas fuera del rango triclase y tipos de etiqueta no enteros.

`src/training/optimizers.py` construye AdamW con dos grupos de parámetros. Las matrices se regularizan mediante decaimiento de pesos; sesgos, parámetros de normalización, token de clasificación y embeddings posicionales quedan excluidos del decaimiento.

### 2.4 Verificación de un paso sintético

`src/training/engine.py` implementa un paso de entrenamiento por lote con:

1. Reinicio de gradientes.
2. Pase hacia adelante.
3. Cálculo y validación de pérdida finita.
4. Retropropagación.
5. Recorte opcional de gradiente.
6. Actualización del optimizador.
7. Salida desacoplada de logits, probabilidades y clases predichas.

Este flujo se prueba con una configuración reducida y tensores sintéticos. Su propósito es validar interfaces y diferenciación automática, no producir pesos entrenados ni resultados clínicos.

## 3. Evidencia verificable

La verificación de S9 se ejecuta desde la raíz del repositorio:

```bash
uv run --locked python scripts/run_s9_verification.py
uv run --locked pytest
```

El primer comando comprueba la arquitectura ViT-B/16 completa, el conteo de parámetros, la forma de salida triclase y un paso sintético con AdamW y pérdida ponderada. La suite unitaria valida configuración, balanceo, optimizador, flujo de gradientes, dimensiones y manejo de errores.

## 4. Criterios y restricciones

La actividad S9 cumple la evidencia de prueba estructural y validación de dimensiones del cronograma. E4 continúa abierto hasta integrar los módulos de datos, configuración, entrenamiento futuro, evaluación e inferencia, además de completar las pruebas de interoperabilidad previstas para S10 y S11.

Las probabilidades obtenidas con parámetros aleatorios no representan predicciones válidas. El proyecto mantiene las exclusiones de la línea base: no entrena el modelo, no genera checkpoints y no calcula AUC, F1, sensibilidad, especificidad ni accuracy sobre datos reales.
