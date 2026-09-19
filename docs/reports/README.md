# Reportes semanales

| Semana | Periodo | Documento final |
| --- | --- | --- |
| S5 | 31 de agosto–6 de septiembre de 2026 | [Reporte S5](Reporte_semanal_S5_final.pdf) |
| S6 | 7–13 de septiembre de 2026 | [Reporte S6](Reporte_semanal_S6_final.pdf) |
| S7 | 14–20 de septiembre de 2026; corte al día 19 | [Reporte S7](Reporte_semanal_S7_final.pdf) |

Los reportes mantienen la organización de indicadores, reporte semanal y control de cambios del formato de referencia S5. S7 registra 24 horas estimadas, distribuidas en 12 por integrante; las horas reales están pendientes de confirmación. La evidencia E2 es sintética.

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
