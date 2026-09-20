# Reportes semanales

| Semana | Periodo | Documento final |
| --- | --- | --- |
| S5 | 31 de agosto–6 de septiembre de 2026 | [Reporte S5](Reporte_semanal_S5_final.pdf) |
| S6 | 7–13 de septiembre de 2026 | [Reporte S6](Reporte_semanal_S6_final.pdf) |
| S7 | 14–20 de septiembre de 2026; corte al día 19 | [Reporte S7](Reporte_semanal_S7_final.pdf) |

Los reportes mantienen la organización de indicadores, reporte semanal y control de cambios del formato de referencia S5. S7 registra 24 horas estimadas, distribuidas en 12 por integrante; las horas reales están pendientes de confirmación. La evidencia E2 es sintética.

## Seguimiento semanal

Cada PDF presenta I1–I10 con resultado de la semana, evidencia y acción de seguimiento. Las definiciones, frecuencias y metas se conservan en la [línea base](../base/Linea_Base_Proyecto_Alzheimer_ViT.pdf); el reporte aplica ese marco al hito correspondiente.

El conteo técnico de I1 usa los cinco componentes EDT de E1 o E2 identificados en cada fila; no equivale a aceptación formal. I2 usa la fecha de aceptación acreditada: E1 tiene constancia del 07/09. I3 deja explícita la discrepancia entre esa fecha y el hito H1 asignado a S5. Los valores declarados sin desglose, las estimaciones y los indicadores sin denominador se identifican como tales.

El semáforo se evalúa con la condición de riesgo alto de la línea base: R7 mantiene exposición 15 mientras no exista cierre o nueva valoración. Por ello se actualiza a amarillo, con acciones específicas por semana. Los originales históricos conservan su estado y contenido anteriores.

## Generación

El contenido está en [weekly_reports.json](../../configs/reports/weekly_reports.json) y la composición en [build_weekly_reports.py](../../scripts/build_weekly_reports.py). Desde la raíz:

```bash
uv run --locked --group reports python scripts/build_weekly_reports.py
```

La orden reemplaza los tres PDF finales. Después de modificar contenido o formato, revisar las páginas renderizadas, los saltos y la legibilidad de las tablas antes de entregar.

## Archivo histórico

- [Original S5](archive/Reporte_semanal_S5_original.pdf): referencia del formato.
- [Original S6](archive/Reporte_semanal_S6_original.pdf): registro previo de avances.
- [Preparación S7](archive/preparacion_s7.md): insumos anteriores a la emisión del reporte final; se conserva como antecedente de elaboración.

Los originales y la línea base se conservan sin alteraciones.
