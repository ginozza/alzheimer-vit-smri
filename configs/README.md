# Configuración

- [e2.json](e2.json): parámetros de geometría, selección de cortes, normalización y límites de procesamiento. Cada corrida guarda su configuración resuelta en `run.json`.
- [e3_architecture.json](e3_architecture.json): dimensiones y parámetros de la arquitectura ViT-B/16 triclase.
- [e4_training.json](e4_training.json): AdamW, pérdida y controles del entrenamiento futuro. Los pesos de clase deben calcularse solo con la partición de entrenamiento.
- [reports/weekly_reports.json](reports/weekly_reports.json): contenido de los reportes S5–S9. El generador produce los PDF en `docs/reports/`.

Las dependencias y herramientas se configuran en `pyproject.toml` en la raíz. Las rutas de los comandos se interpretan desde esa raíz. Los cambios metodológicos deben contrastarse con la línea base; no ajustar parámetros contra el conjunto de test.
