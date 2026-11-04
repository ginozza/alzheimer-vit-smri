# Configuración

- [e2.json](e2.json): parámetros de geometría, selección de cortes, normalización y límites de procesamiento. Cada corrida guarda su configuración resuelta en `run.json`.
- [e3_architecture.json](e3_architecture.json): dimensiones y parámetros de la arquitectura ViT-B/16 triclase.
- [e4_training.json](e4_training.json): AdamW, pérdida y controles del entrenamiento futuro. Los pesos de clase deben calcularse solo con la partición de entrenamiento.
- [e4_dataset.json](e4_dataset.json): tamaño de lote, subprocesos y contrato tensorial de carga de datos.
- [e4_inference.json](e4_inference.json): parámetros del pipeline de inferencia triclase.
- [e5_interpretabilidad.json](e5_interpretabilidad.json): hiperparámetros del algoritmo Attention Rollout e interpolación espacial.
- [e6_integration.json](e6_integration.json): parámetros del flujo integrado E6, banderas de modelo no entrenado y umbrales del benchmark de portabilidad I9.
- [reports/weekly_reports.json](reports/weekly_reports.json): contenido de los reportes S5–S14. El generador produce los PDF en `docs/reports/`.

Las dependencias y herramientas se configuran en `pyproject.toml` en la raíz. Las rutas de los comandos se interpretan desde esa raíz. Los cambios metodológicos deben contrastarse con la línea base; no ajustar parámetros contra el conjunto de test.
