# TP2 — Reconstrucción 3D

Notebook principal: `tp2_reconstruction.ipynb`.

Se ejecuta desde la raíz del proyecto, donde debe existir `i308-calib/data/` con `captures/`, `stereo_calibration.pkl` y `stereo_maps.pkl`. No requiere configuración YAML.

Instalación: `pip install -r requirements.txt`. Todas las constantes experimentales están definidas en la celda **0** del notebook.

Ejecutar las celdas en orden. La selección inicial son las capturas `0..13`, que dieron mejor resultado visual en las pruebas. Los archivos se exportan en `outputs/` (`preview_roi.ply`, `preview_clean.ply`, `preview_object.ply`, `preview_clustered.ply`, `shoe_final.ply`, `camera_poses.png`, `measurement.png`).

La medición es una estimación sobre cobertura parcial del zapato, con percentiles 1–99 y una altura referida al plano del tablero ChArUco.
