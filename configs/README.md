# Configuración

- [e2.json](e2.json): parámetros de geometría, selección de cortes, normalización y límites de procesamiento. Cada corrida guarda su configuración resuelta en `run.json`.
- [reports/weekly_reports.json](reports/weekly_reports.json): contenido de los reportes S5–S7. El generador produce los PDF en `docs/reports/`.

Las dependencias y herramientas se configuran en `pyproject.toml` en la raíz. Las rutas de los comandos se interpretan desde esa raíz. Los cambios metodológicos deben contrastarse con la línea base; no ajustar parámetros contra el conjunto de test.
