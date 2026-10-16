from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoardPose:
    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray
    ids: np.ndarray
    rmse: float


def estimate_pose(
    image: np.ndarray,
    detector: cv2.aruco.CharucoDetector,
    K: np.ndarray,
    dist: np.ndarray,
) -> BoardPose | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None

    # Los IDs identifican las coordenadas 3D conocidas, incluso con oclusiones.
    world_points = detector.getBoard().getChessboardCorners()[ids.flatten()]
    image_points = corners.reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(world_points, image_points, K, dist)
    if not success:
        return None

    projected, _ = cv2.projectPoints(world_points, rvec, tvec, K, dist)
    residuals = projected.reshape(-1, 2) - image_points
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return BoardPose(rvec, tvec, corners, ids, rmse)


def world_from_rectified(
    pose: BoardPose, R1: np.ndarray, T_world_board: np.ndarray
) -> np.ndarray:
    """Transformación 4x4: cámara rectificada -> mundo (Z positivo sobre el tablero)."""
    R, _ = cv2.Rodrigues(pose.rvec)
    R_inv = R.T

    transform = np.eye(4)
    transform[:3, :3] = R_inv @ R1.T
    transform[:3, 3] = -R_inv @ pose.tvec.reshape(3)
    return T_world_board @ transform
