import pickle
from pathlib import Path

import cv2


def load_calibration(calibration_path: Path, maps_path: Path) -> tuple[dict, dict]:
    with calibration_path.open('rb') as f:
        calibration = pickle.load(f)

    with maps_path.open('rb') as f:
        maps = pickle.load(f)

    return calibration, maps


def rectify_pair(left, right, maps):
    left_rect = cv2.remap(
        left, maps['left_map_x'], maps['left_map_y'], cv2.INTER_LINEAR
    )
    right_rect = cv2.remap(
        right, maps['right_map_x'], maps['right_map_y'], cv2.INTER_LINEAR
    )
    return left_rect, right_rect
