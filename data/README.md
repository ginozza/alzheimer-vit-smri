# Datos

`inventory/` conserva los CSV ADNI/OASIS, el diccionario YAML y el resumen de cohortes originales. Son referencias de metadatos; no incluyen volúmenes NIfTI ni acreditan su preparación anatómica.

`raw/` es la ubicación prevista para imágenes locales autorizadas. `processed/` contiene corridas independientes con sus entradas de referencia, manifiestos, registros y tensores. Ambas ubicaciones están excluidas de Git. Los manifiestos locales `local_*.csv` también se excluyen.

Se conserva `processed/e2_demo_verified/`, una corrida exclusivamente sintética disponible solo en este entorno local; no forma parte de Git. Cada nueva ejecución debe usar otra carpeta; el pipeline rechaza salidas existentes.

No hay imágenes reales ADNI/OASIS disponibles. Antes de incorporarlas, cumplir el [contrato de entrada E2](../scripts/README.md#contrato-de-entrada-e2), conservar las particiones por sujeto y verificar la procedencia del registro MNI152 y de la extracción cerebral.
