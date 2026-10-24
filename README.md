# TP2 — Reconstrucción 3D estéreo

Reconstrucción de una nube de puntos 3D de un zapato a partir de imágenes estéreo calibradas. El proyecto compara StereoSGBM con CREStereo y utiliza un tablero ChArUco para ubicar las capturas en un marco global y medir el objeto en milímetros.

## Estructura

```text
.
├── data/
│   ├── calib/                  # 14 pares de calibración
│   ├── captures/               # 28 pares estéreo del objeto
│   ├── stereo_calibration.pkl  # Parámetros de calibración
│   └── stereo_maps.pkl         # Rectificación y matriz Q
├── src/                        # Funciones del pipeline, incluido stereo_neural.py
├── tp2_reconstruction.ipynb   # Procesamiento, experimentos y visualizaciones
├── requirements.txt
├── models/                     # Modelo ONNX descargado al ejecutar
└── outputs/                    # Figuras y nubes generadas localmente
```

El repositorio `i308-calib` se utilizó como referencia para la calibración, pero no es una dependencia del código.

## Instalación

La ejecución documentada se realizó con **Python 3.14.7**. Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install onnxruntime jupyterlab
```

En Windows, activar el entorno con `.venv\Scripts\Activate.ps1`. `requirements.txt` contiene NumPy, OpenCV Contrib, Matplotlib, Open3D, nbformat y SciPy.

## Datos

Colocar los pares `left_N.jpg` y `right_N.jpg` en `data/captures/` y `data/calib/`, según corresponda. Las imágenes tienen una resolución de **1280 × 720 píxeles**. Los archivos `.pkl` deben corresponder al mismo sistema estéreo y estar ubicados en `data/`.

El dataset se distribuye por separado del código cuando su tamaño lo requiera.

## Ejecución

```bash
jupyter lab tp2_reconstruction.ipynb
```

Ejecutar las celdas en orden desde la raíz del proyecto. El notebook:

1. Carga y verifica la calibración, rectifica las imágenes y evalúa StereoSGBM con los experimentos A–D.
2. Estima las poses con ChArUco y transforma las nubes al marco global del tablero.
3. Reconstruye con **CREStereo** las **27 capturas con pose válida**. Si falta el modelo, lo descarga en `models/`; la primera ejecución requiere conexión a Internet.
4. Aplica ROI, voxelización de **3 mm**, soporte de al menos **tres vistas adicionales** a **6 mm**, SOR (**20 vecinos**, `std_ratio=1.5`) y voxelización final.
5. Calcula las dimensiones mediante PCA y percentiles 1–99, tomando el tablero como referencia de altura.

## Resultado y entrega

La nube definitiva se exporta como **`outputs/shoe_final_dl.ply`**, en formato PLY binario con colores.

- **Puntos:** 24.037
- **Dimensiones (largo × ancho × alto):** 283,6 × 92,7 × 114,0 mm

Las cifras corresponden a la ejecución documentada en el notebook. Para la entrega se proporciona `shoe_final_dl.ply` como archivo independiente, junto con el informe y el código fuente.
