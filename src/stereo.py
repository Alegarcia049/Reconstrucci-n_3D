import cv2
import numpy as np


def compute_disparity(
    left: np.ndarray,
    right: np.ndarray,
    min_disparity: int,
    num_disparities: int,
    block_size: int,
    uniqueness_ratio: int,
) -> np.ndarray:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        uniquenessRatio=uniqueness_ratio,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def reproject_to_3d(
    disparity: np.ndarray,
    left: np.ndarray,
    Q: np.ndarray,
    min_disparity: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.reprojectImageTo3D(disparity, Q)
    valid = (disparity > min_disparity) & np.isfinite(points).all(axis=2)
    colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    return points[valid], colors[valid]
