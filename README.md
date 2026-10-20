# TP2 — Reconstrucción 3D mediante visión estéreo

Trabajo práctico de **Visión Artificial (I308, Universidad de San Andrés)**. Se reconstruye una nube de puntos 3D de un zapato a partir de pares de imágenes estéreo calibradas, utilizando un tablero **ChArUco** como referencia geométrica y métrica.

El proyecto implementa el procesamiento en Python con **OpenCV, NumPy, SciPy y Open3D**. El notebook principal permite inspeccionar las etapas intermedias, comparar estrategias de filtrado y exportar la nube final en formato PLY.

## Objetivo y alcance

El procedimiento parte de imágenes izquierda/derecha y recupera puntos tridimensionales mediante disparidad estéreo. Cada captura se ubica en un sistema de coordenadas común a través de la pose del tablero; posteriormente, se fusionan y filtran las nubes parciales para obtener una representación del zapato y estimar sus dimensiones en milímetros.

El resultado es una **nube de puntos coloreada**, no una malla cerrada. La medición depende de las superficies visibles, la calidad de las correspondencias estéreo y las decisiones de filtrado; no debe interpretarse como una medición metrológica certificada.

## Estructura del proyecto

```text
.
├── data/                      # Datos del TP2, dentro del repositorio principal
│   ├── calib/                 # 14 pares de calibración (checkerboard)
│   ├── captures/              # 28 pares estéreo del zapato
│   ├── stereo_calibration.pkl # Parámetros intrínsecos y extrínsecos
│   └── stereo_maps.pkl        # Mapas de rectificación y matriz Q
├── i308-calib/                # Submódulo de referencia (código de calibración)
├── src/
│   ├── calibration.py         # Carga de calibración y rectificación
│   ├── dataset.py             # Lectura y asociación de pares estéreo
│   ├── stereo.py              # Disparidad SGBM y reproyección 3D
│   ├── pose.py                # Pose ChArUco y transformaciones rígidas
│   ├── pointcloud.py          # Nubes, recortes y medición
│   ├── evaluation.py          # Diagnósticos geométricos
│   └── visualization.py       # Gráficos, poses y geometría de cámaras
├── tp2_reconstruction.ipynb   # Pipeline y experimentos
├── requirements.txt
└── outputs/                   # Nubes y figuras generadas localmente
```

`data/` pertenece al **repositorio principal** y contiene los archivos que consume el notebook. El submódulo `i308-calib/` se mantiene para consultar y reproducir el procedimiento original de calibración; **ningún módulo de `src/` ni el notebook importa funciones de él**. Los resultados de `outputs/` se generan localmente.

## Instalación

Se recomienda trabajar con **Python 3.14** y un entorno virtual. Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scipy jupyterlab
```

En Windows, la activación equivalente en PowerShell es `.venv\Scripts\Activate.ps1`. El último comando instala las dependencias que utiliza el diagnóstico multivista (`scipy.spatial.cKDTree`) y la ejecución del notebook (`jupyterlab`), porque no figuran en el `requirements.txt` de la versión revisada.

**No es necesario inicializar `i308-calib` para ejecutar el TP2**: el código propio lee los pares de imágenes y los archivos `.pkl` directamente desde `data/`. Solo si se desea consultar o volver a ejecutar el código original de calibración, puede obtenerse el submódulo con:

```bash
git submodule update --init --recursive
```

Para ejecutar la reconstrucción tampoco hace falta volver a calibrar, siempre que `data/stereo_calibration.pkl` y `data/stereo_maps.pkl` correspondan al mismo sistema de cámaras, resolución y configuración óptica que las capturas.

## Datos de entrada

El notebook espera los siguientes archivos:

| Ruta                            | Contenido                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `data/captures/left_N.jpg`    | Imagen izquierda de la captura `N`                                            |
| `data/captures/right_N.jpg`   | Imagen derecha de la misma captura                                              |
| `data/stereo_calibration.pkl` | Intrínsecos, distorsión y transformación relativa de las cámaras            |
| `data/stereo_maps.pkl`        | Mapas de rectificación y matrices `R1`, `R2`, `P1`, `P2`, `Q`        |
| `data/calib/`                 | Imágenes del checkerboard utilizadas para evaluar o reproducir la calibración |

El conjunto de trabajo tiene **28 pares estéreo del objeto** y **14 pares de calibración**, con imágenes de **1280 × 720 px por cámara**. La calibración original utiliza un **checkerboard convencional**; en cambio, la localización de las capturas del objeto utiliza un **ChArUco de 9 × 13 cuadrados**, con lado de **42 mm** y marcadores de **31,5 mm**.

`data/calib/` se usa para reproducir la evaluación del RMS de calibración al inicio del notebook; para reconstruir el zapato con parámetros ya calculados bastan `data/captures/` y los dos archivos `.pkl`. Ninguno de estos recursos tiene que estar dentro del submódulo.

> Los archivos pickle deben provenir de una fuente confiable. Además, sus parámetros deben corresponder a las cámaras, la resolución y la configuración óptica de las fotografías utilizadas.

## Ejecución

Iniciar Jupyter desde la raíz del proyecto:

```bash
jupyter lab tp2_reconstruction.ipynb
```

Ejecutar las celdas **en orden**, comenzando por la sección de imports y configuración. En la celda de rutas, usar:

```python
DATA_DIR = Path("data")
CAPTURE_DIR = DATA_DIR / "captures"
CALIBRATION_PATH = DATA_DIR / "stereo_calibration.pkl"
MAPS_PATH = DATA_DIR / "stereo_maps.pkl"
```

Estas rutas reemplazan las antiguas referencias a `i308-calib/data/`. Los parámetros experimentales se definen en el propio notebook; **no se utiliza un archivo YAML de configuración** para esta reconstrucción.

Para reproducir la configuración con 21 capturas y filtrado multivista elegida en los últimos experimentos, seleccionar:

```python
EXPERIMENT = "D"
SELECTED_VIEW_IDS = VIEW_SETS[EXPERIMENT]
MIN_OTHER_VIEWS = 3
```

En la versión del notebook donde `MIN_OTHER_VIEWS` se define en la sección de soporte multivista, modificarlo allí. Las salidas se guardan en `outputs/<EXPERIMENT>/` si `OUTPUT_DIR` utiliza esa convención.

## Pipeline de reconstrucción

1. **Calibración y rectificación.** Se cargan desde `data/` los parámetros generados originalmente con [`i308-calib`](https://github.com/udesa-vision/i308-calib); la rectificación se realiza con código propio. `cv2.remap()` aplica los mapas de rectificación para alinear las líneas epipolares horizontalmente.
2. **Disparidad.** `StereoSGBM` estima la diferencia horizontal entre puntos correspondientes. La profundidad está relacionada con la disparidad mediante `Z = f·B/d`.
3. **Reproyección 3D.** `cv2.reprojectImageTo3D()` emplea la matriz `Q` para recuperar los puntos en el marco de la cámara izquierda rectificada.
4. **Estimación de poses.** Se detectan esquinas ChArUco y se estima la pose de cada captura con `cv2.solvePnP()`.
5. **Transformación y fusión.** Las nubes se llevan al marco global del tablero, se recortan espacialmente y se reducen mediante vóxeles de 3 mm.
6. **Filtrado del objeto.** Se comparan alternativas estadísticas y un criterio de **consistencia multivista**: cada punto debe tener un vecino a no más de 6 mm en al menos **tres vistas adicionales**.
7. **Medición y exportación.** Se obtiene la orientación principal mediante PCA en XY; las dimensiones se calculan usando percentiles robustos y la altura se refiere al plano de apoyo conocido.

### Sistema de coordenadas

Todas las poses y nubes exportadas utilizan un marco global común:

- El tablero ChArUco ocupa el plano **Z = 0**.
- El eje **+Z** apunta hacia el zapato.
- Las coordenadas y dimensiones están expresadas en **milímetros**.

La transformación desde la cámara izquierda rectificada combina la rotación `R1`, la pose PnP y el cambio de marco global. Esto permite fusionar capturas tomadas desde distintas posiciones sin confundir los marcos locales de las cámaras.

## Experimentos y criterios de selección

Se compararon cuatro conjuntos de capturas:

| Experimento | Selección                     | Propósito                                                           |
| ----------- | ------------------------------ | -------------------------------------------------------------------- |
| **A** | 0–13 (14 vistas)              | Reconstrucción de referencia                                        |
| **B** | 0–13, sin la 4 (13 vistas)    | Evaluar la exclusión de una captura con errores extremos            |
| **C** | 0–14, sin la 4 (14 vistas)    | Sustituir la vista 4 manteniendo la cantidad de capturas             |
| **D** | 0–22, sin 4 ni 19 (21 vistas) | Aumentar cobertura con vistas de mejor consistencia sobre el tablero |

Las métricas del tablero sirven para **diagnosticar** las vistas, pero no garantizan una reconstrucción correcta del zapato. Por eso se evaluaron también las nubes resultantes.

Se probaron **SOR global + DBSCAN**, **SOR localizado + filtro por radio + DBSCAN** y **soporte multivista** con requisitos de 1, 2 y 3 vistas adicionales. Los filtros basados solo en densidad local podían conservar estructuras espurias densas. El soporte multivista ≥3 produjo una mejora visual importante; aplicar SOR adicional introdujo huecos y redujo la cobertura superficial. Por esa razón se eligió el soporte multivista como filtrado final para D.

Los parámetros de ROI, vóxel, tolerancia y percentiles son **heurísticos**. La selección de vistas y el número mínimo de cámaras de soporte se estudiaron comparativamente; no se realizó una optimización exhaustiva de todos los hiperparámetros.

## Evaluación

El notebook incluye diagnósticos con significados diferentes:

| Métrica                     | Qué evalúa                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------ |
| RMS de calibración estéreo | Consistencia del ajuste sobre el checkerboard de calibración                  |
| RMSE epipolar vertical       | Alineación de correspondencias entre imágenes rectificadas                   |
| RMSE de PnP                  | Error de reproyección de las esquinas ChArUco de cada captura                 |
| Residual de disparidad       | Diferencia entre la disparidad SGBM y la predicción geométrica basada en PnP |
| Distancia al plano ChArUco   | Desviación de los puntos reconstruidos del tablero respecto de `Z = 0`      |
| Solapamiento entre vistas    | Proximidad espacial entre nubes de distintas capturas                          |
| Sensibilidad de dimensiones  | Variación del largo, ancho y alto al modificar los percentiles de recorte     |

Estas métricas **no son intercambiables**. En particular, un RMSE bajo de pose no asegura una buena estimación de disparidad sobre el zapato; y un mayor número de puntos no implica una geometría más exacta.

## Resultados de referencia

En una ejecución del **experimento D** con soporte multivista en al menos tres vistas adicionales se obtuvieron aproximadamente:

| Resultado                  |              Valor |
| -------------------------- | -----------------: |
| Puntos de la nube filtrada |   **29.009** |
| Largo estimado             | **284,6 mm** |
| Ancho estimado             |  **96,7 mm** |
| Alto estimado              | **114,3 mm** |

Estos valores corresponden a un resultado experimental particular, con recorte por percentiles **1–99**. No representan dimensiones reales verificadas con un instrumento externo. La exactitud de la nube no puede determinarse únicamente a partir de la estabilidad de estas medidas.

## Archivos generados

El notebook puede producir, dentro de `outputs/<EXPERIMENT>/`:

| Archivo                                                                  | Descripción                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `shoe_final.ply`                                                       | Nube del zapato elegida para la exportación final, en PLY binario |
| `preview_roi.ply`                                                      | Nube fusionada recortada y voxelizada                              |
| `preview_clean.ply`, `preview_object.ply`, `preview_clustered.ply` | Resultados del pipeline estadístico de referencia                 |
| `preview_local_clean.ply`                                              | Resultado del filtrado local experimental                          |
| `preview_multiview.ply`                                                | Nube con el umbral multivista seleccionado                         |
| `preview_support_1.ply`, `preview_support_2.ply`                     | Otras variantes de soporte, si se ejecutan                         |
| `preview_by_view.ply`                                                  | Nubes coloreadas artificialmente según la captura de origen       |
| `camera_poses.png`, `camera_poses.ply`                               | Visualización de posiciones y orientaciones de cámaras           |
| `measurement.png`                                                      | Nube con caja orientada y dimensiones estimadas                    |

Los resultados de evaluación se **imprimen en las celdas**. No se necesitan archivos CSV para ejecutar ni comprender los experimentos.

Para explorar la geometría en 3D, importar `shoe_final.ply` y, opcionalmente, `camera_poses.ply` como **capas separadas en MeshLab**. El segundo archivo representa las cámaras mediante frustums esquemáticos y no forma parte de la nube final del objeto.

## Limitaciones

- La disparidad estéreo es sensible a zonas sin textura, reflejos, oclusiones y límites de profundidad.
- El RMS de calibración y el RMSE de PnP son errores de reproyección; **no equivalen** al error métrico absoluto de la nube.
- Un filtro multivista estricto puede descartar superficies reales observadas desde pocas cámaras.
- El recorte próximo al tablero elimina posibles puntos de la suela. Por eso, la altura se estima respecto del plano de apoyo `Z = 0`.
- Sin un modelo 3D o mediciones físicas independientes del zapato, las dimensiones y la calidad superficial no pueden validarse como exactitud absoluta.

## Herramientas utilizadas

Python · OpenCV (incluido `aruco`) · NumPy · SciPy (`cKDTree`) · Open3D · Matplotlib · Jupyter. La calibración original se realizó mediante [`i308-calib`](https://github.com/udesa-vision/i308-calib), que **no es una dependencia de ejecución ni una librería importada por el proyecto**.
