# Informe de Avance Semanal — Semana 14 (S14)

**Proyecto:** Detección prodrómica de la enfermedad de Alzheimer mediante Vision Transformers aplicados a imágenes de resonancia magnética estructural (`alzheimer-vit-smri`)  
**Autores:** Malak Sánchez y Juan Simancas  
**Director:** Sergio Lubo  
**Institución:** Universidad del Magdalena — Programa de Ingeniería de Sistemas (Seminario III, 2026-II)  
**Periodo:** Semana 14 | 02 de noviembre a 08 de noviembre de 2026  
**Paquete EDT:** EDT 7.0 — Integración (Componentes 7.2 Interfaz, 7.4 Manual y 7.5 Demostración)  
**Entregable asociado:** Entregable E6: Prototipo de integración (**Aceptado Formalmente — Hito H6**)  
**Estado general:** Verde  
**PDF oficial:** [`Reporte_semanal_S14_final.pdf`](Reporte_semanal_S14_final.pdf)

---

## 1. Objetivos

De acuerdo con la planificación oficial establecida en la **Línea Base del Proyecto** (`docs/base/Linea_Base_Proyecto_Alzheimer_ViT.pdf`, Secciones 5, 6, 7 y 8):

1. **Objetivo del Cronograma (Semana 14):** *«Pruebas, corrección de errores y elaboración del manual de uso»*, alcanzando el **Hito H6: E6 aceptado** bajo responsabilidad conjunta de ambos integrantes.
2. **Cumplimiento del Criterio de Aceptación Oficial del Entregable E6 (Sección 6, pág. 4):**
   > *«La interfaz recibe una entrada de prueba, ejecuta el flujo técnico simulado, controla errores e indica que no contiene un modelo entrenado ni genera diagnósticos.»*
3. **Componentes EDT 7.0 a consolidar en S14:**
   - **EDT 7.2 (Interfaz):** Completar la interfaz técnica local (programática y CLI) conectada al flujo simulado, con captura controlada de errores y avisos explícitos.
   - **EDT 7.4 (Manual):** Elaborar el Manual Técnico y de Uso del sistema (instalación, configuración, ejecución del flujo simulado, catálogo de errores y procedimiento de entrenamiento futuro).
   - **EDT 7.5 (Demostración):** Proveer un script de demostración y auditoría técnica repetible (`scripts/run_e6_verification.py`).

---

## 2. Contexto

Al comenzar la Semana 14, el proyecto contaba con los entregables **E1, E2, E3, E4 y E5 formalmente aceptados (71,43 %)** y con el avance técnico de la **Semana 13** completado:
- El flujo integrado completo (`IntegratedSystemPipeline` en `src/integration/pipeline.py`) ya enlazaba la generación/validación de tripletes 2.5D $(3, 224, 224)$ `float32`, el modelo `VisionTransformer` ViT-B/16 triclase, `InferencePipeline` y `AttentionRollout`.
- El **Indicador I9 (Portabilidad)** había sido medido y documentado en S13 (327,30 MB en RAM, ~585–601 ms en CPU y 0,00 % de degradación en exportación TorchScript).
- La suite acumulada registraba **167 pruebas automatizadas aprobadas**.

Para cerrar formalmente el **Entregable E6 (Hito H6)** en S14 restaba consolidar la interfaz técnica de usuario (`TechnicalIntegrationInterface` y su CLI), validar exhaustivamente el control de errores frente a entradas defectuosas, elaborar el Manual de Uso y actualizar la documentación transversal del repositorio.

---

## 3. Actividades realizadas

1. **Implementación de la interfaz técnica local y CLI (`src/integration/interface.py`):**
   - Desarrollo de la clase `TechnicalIntegrationInterface` con soporte para tres modalidades de entrada de prueba: generación sintética controlada por clase y semilla (`input_data=None`), ruta a archivo `.npy` en disco (`str` / `Path`) y tensor en memoria (`np.ndarray` / `torch.Tensor`).
   - Implementación del formateador de reporte técnico (`format_execution_report`) y del punto de entrada por línea de comandos (`build_cli_parser`, `run_cli`) con códigos de retorno deterministas (`0` en ejecución exitosa, `1` ante error de entrada controlado).
2. **Endurecimiento y validación del control estructurado de errores (`IntegrationInputError`):**
   - Verificación sistemática de 11 códigos de error controlados: `FILE_NOT_FOUND`, `UNSUPPORTED_FILE_EXTENSION`, `CORRUPT_NPY_FILE`, `INVALID_INPUT_TYPE`, `EMPTY_INPUT_TENSOR`, `NON_FINITE_VALUES`, `INVALID_DIMENSIONS`, `UNSUPPORTED_BATCH_SIZE`, `SHAPE_MISMATCH`, `DEGENERATE_CONSTANT_SIGNAL` e `INVALID_TARGET_CLASS`.
3. **Incorporación obligatoria de avisos de modelo no entrenado y no diagnóstico (Riesgos R6 y R7):**
   - Inclusión en `IntegrationExecutionResult`, en los archivos JSON exportados y en la salida de consola de las banderas `is_trained_model=False`, `generates_clinical_diagnosis=False`, del aviso `UNTRAINED_MODEL_NOTICE` y del descargo `CLINICAL_DISCLAIMER`.
4. **Elaboración del Manual Técnico y de Uso y del documento de aceptación de E6:**
   - Redacción de `docs/deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md` con instrucciones reproducibles de instalación (`uv sync --locked`), descripción de los 7 archivos JSON de configuración, guía de uso CLI y programática, catálogo de errores y protocolo paso a paso para entrenamiento futuro.
   - Actualización de `README.md`, `docs/README.md`, `configs/README.md`, `scripts/README.md` y `docs/reports/README.md`.
5. **Desarrollo del runner de auditoría formal H6 y suite de pruebas S14:**
   - Creación de `scripts/run_e6_verification.py` y `tests/test_e6_interface_and_error_handling.py` (16 pruebas nuevas).

---

## 4. Implementación

| Archivo / Módulo | Tipo | Descripción Técnica |
|---|---|---|
| [`src/integration/interface.py`](../../src/integration/interface.py) | Creado | `TechnicalIntegrationInterface`, `format_execution_report`, `build_cli_parser` y `run_cli`. |
| [`scripts/run_e6_verification.py`](../../scripts/run_e6_verification.py) | Creado | Runner de auditoría formal de los criterios de aceptación del Entregable E6 (Hito H6). |
| [`tests/test_e6_interface_and_error_handling.py`](../../tests/test_e6_interface_and_error_handling.py) | Creado | 16 pruebas unitarias y de integración sobre interfaz, CLI, control de errores y avisos obligatorios. |
| [`docs/deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md`](../deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md) | Creado | Documento formal de aceptación del Entregable E6 y Manual Técnico y de Uso. |
| [`configs/reports/weekly_reports.json`](../../configs/reports/weekly_reports.json) | Modificado | Incorporación de los registros estructurados de las Semanas 13 y 14 (I1–I10, reporte y cambios). |
| [`docs/reports/Reporte_semanal_S13_final.pdf`](Reporte_semanal_S13_final.pdf) | Creado | Reporte oficial de la Semana 13 en formato PDF de 3 páginas. |
| [`docs/reports/Reporte_semanal_S14_final.pdf`](Reporte_semanal_S14_final.pdf) | Creado | Reporte oficial de la Semana 14 en formato PDF de 3 páginas. |
| [`README.md`](../../README.md), [`docs/README.md`](../README.md), [`configs/README.md`](../../configs/README.md), [`scripts/README.md`](../../scripts/README.md), [`docs/reports/README.md`](README.md) | Modificados | Sincronización documental hasta la Semana 14 (Entregables E1–E6 aceptados). |

---

## 5. Decisiones técnicas

1. **Separación entre modo seguro de interfaz (`safe_mode=True`) y modo de excepción programática (`safe_mode=False`):** En `TechnicalIntegrationInterface`, todas las llamadas operan con `safe_mode=True` para garantizar que cualquier anomalía en los datos de entrada sea capturada y reportada dentro de `IntegrationExecutionResult(status="ERROR", error_code=...)` sin abortar el proceso con trazas no controladas.
2. **Persistencia de avisos metodológicos incluso en respuestas de error:** Se decidió que `IntegrationExecutionResult` incluya `UNTRAINED_MODEL_NOTICE` y `CLINICAL_DISCLAIMER` tanto cuando `status == "SUCCESS"` como cuando `status == "ERROR"`, asegurando que ninguna salida de la interfaz pueda interpretarse fuera de su contexto académico.
3. **Alineación dimensional estricta $(3, 224, 224)$ en las pruebas de interfaz:** Dado que `InferencePipeline` (E4) exige por contrato tensores 2.5D de $3 \times 224 \times 224$, los fixtures rápidos de prueba mantienen la resolución espacial canónica $224 \times 224$ y reducen únicamente la profundidad (`depth=2`) y dimensión de embedding (`embed_dim=48`).

---

## 6. Resultados

La auditoría formal del Entregable E6 (`uv run --locked python scripts/run_e6_verification.py`) confirmó el cumplimiento íntegro del criterio de aceptación:

| Auditoría | Resultado Verificado |
|---|---|
| **Auditoría 1 — Entradas de prueba y flujo simulado** | Interfaz probada con entrada sintética (`MCI`), archivo `.npy` externo (`AD`) e invocación CLI (`CN`, código de salida `0`), exportando `.npy`, `heatmap.png`, `overlay.png` y `result.json`. |
| **Auditoría 2 — Control de errores** | 8 escenarios de error en el runner y 11 en la suite de pruebas capturados limpiamente con códigos `FILE_NOT_FOUND`, `UNSUPPORTED_FILE_EXTENSION`, `CORRUPT_NPY_FILE`, `INVALID_DIMENSIONS`, `UNSUPPORTED_BATCH_SIZE`, `SHAPE_MISMATCH`, `NON_FINITE_VALUES`, `DEGENERATE_CONSTANT_SIGNAL` e `INVALID_TARGET_CLASS`. |
| **Auditoría 3 — Avisos obligatorios E6 y R7** | Verificada la presencia de `is_trained_model=False`, `generates_clinical_diagnosis=False`, `UNTRAINED_MODEL_NOTICE` y `CLINICAL_DISCLAIMER` en tiempo de ejecución, JSON y salida de consola. |
| **Auditoría 4 — Manual de uso y documentación** | Verificada la existencia e integridad de `docs/deliverables/E6_Prototipo_Integracion_y_Manual_de_Uso.md` y `docs/implementation/E6_S13_Flujo_Integrado_y_Portabilidad.md`. |
| **Avance acumulado de entregables (Indicador I2)** | **85,71 % (6/7 entregables formalmente aceptados: E1, E2, E3, E4, E5 y E6)**. |

---

## 7. Validaciones

1. **Ejecución del runner de aceptación E6 (`scripts/run_e6_verification.py`):** **SUPERADO AL 100 %** (0 errores, código de salida `0`).
2. **Ejecución de pruebas de S13 y S14 (`tests/test_s13_integration_and_portability.py` y `tests/test_e6_interface_and_error_handling.py`):** **26/26 pruebas aprobadas**.
3. **Ejecución de la suite completa del repositorio (`uv run --locked pytest`):** **183/183 pruebas aprobadas al 100 %** sin regresiones en ningún módulo previo (E1 a E5).

---

## 8. Dificultades

1. **Validación de resolución fija en `InferencePipeline._prepare_tensor`:** Se ajustaron los fixtures de prueba al tamaño contractual $(3, 224, 224)$ con un codificador ligero (`embed_dim=48, depth=2`), garantizando pruebas rápidas y estricta compatibilidad dimensional.

---

## 9. Limitaciones

1. **Interfaz local de línea de comandos y API programática:** Conforme a la Línea Base (Sección 2 y 3.2), la interfaz es una interfaz técnica local de integración y demostración; no incluye servidor web en la nube ni integración con sistemas hospitalarios PACS/HIS.
2. **Modelo sin entrenamiento clínico:** De acuerdo con las exclusiones semestrales, el prototipo opera con pesos inicializados sin entrenar y no debe emplearse para inferencia clínica real.

---

## 10. Conclusiones

1. Se cumplieron al **100 %** los objetivos de la **Semana 14**, alcanzando formalmente el **Hito H6 (Entregable E6: Prototipo de integración aceptado)**.
2. El sistema cuenta ahora con **6 de los 7 entregables semestrales completados y aceptados (85,71 %)**, respaldados por **183 pruebas automatizadas aprobadas**, manual técnico de uso y reportes semanales en PDF y Markdown.

---

## 11. Trabajo futuro

Para las **Semanas 15 y 16 (Paquete EDT 8.0 — Cierre del proyecto / Entregable E7)** restan las actividades finales del cronograma:
1. **Semana 15:** Depuración final del repositorio, consolidación integral de resultados y trazabilidad, elaboración del informe final del proyecto y preparación de la presentación.
2. **Semana 16:** Entrega final, sustentación académica, demostración repetible y cierre formal del proyecto (**Hito H7: E7 aceptado**).
