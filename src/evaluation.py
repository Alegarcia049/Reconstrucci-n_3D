import cv2
import numpy as np

from src.pose import world_from_rectified


def epipolar_vertical_errors(left_corners, left_ids, right_corners, right_ids,
                             calibration, maps):
    """Residuos firmados v_L - v_R para esquinas ChArUco con IDs compartidos."""
    left_by_id = dict(zip(left_ids.ravel(), left_corners.reshape(-1, 2)))
    right_by_id = dict(zip(right_ids.ravel(), right_corners.reshape(-1, 2)))
    shared_ids = sorted(left_by_id.keys() & right_by_id.keys())

    if not shared_ids:
        return np.array([], dtype=np.float32)

    left = np.asarray([left_by_id[i] for i in shared_ids], np.float32).reshape(-1, 1, 2)
    right = np.asarray([right_by_id[i] for i in shared_ids], np.float32).reshape(-1, 1, 2)

    left_rect = cv2.undistortPoints(
        left, calibration["left_K"], calibration["left_dist"],
        R=maps["R1"], P=maps["P1"][:, :3],
    )
    right_rect = cv2.undistortPoints(
        right, calibration["right_K"], calibration["right_dist"],
        R=maps["R2"], P=maps["P2"][:, :3],
    )
    return left_rect[:, 0, 1] - right_rect[:, 0, 1]


def board_stereo_errors(disparity, pose, board, calibration, maps,
                        T_world_board, min_disparity, patch_radius=2):
    """Residuos de disparidad y de altura respecto de Z_w=0 para esquinas ChArUco.

    La mediana de un parche evita consultar una única disparidad sobre una esquina.
    Los errores se condicionan a las esquinas para las cuales hay disparidad válida.
    """
    board_xyz = board.getChessboardCorners()[pose.ids.ravel()]
    R, _ = cv2.Rodrigues(pose.rvec)
    camera_xyz = board_xyz @ R.T + pose.tvec.reshape(3)
    rect_xyz = camera_xyz @ maps["R1"].T
    hom_xyz = np.column_stack((rect_xyz, np.ones(len(rect_xyz))))

    p_left = hom_xyz @ maps["P1"].T
    p_right = hom_xyz @ maps["P2"].T
    disparity_expected = p_left[:, 0] / p_left[:, 2] - p_right[:, 0] / p_right[:, 2]

    rect_corners = cv2.undistortPoints(
        pose.corners, calibration["left_K"], calibration["left_dist"],
        R=maps["R1"], P=maps["P1"][:, :3],
    ).reshape(-1, 2)
    T_world_rect = world_from_rectified(pose, maps["R1"], T_world_board)

    height, width = disparity.shape
    errors_d, errors_plane = [], []
    for (u, v), expected in zip(rect_corners, disparity_expected):
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height):
            continue
        patch = disparity[
            max(0, y - patch_radius): min(height, y + patch_radius + 1),
            max(0, x - patch_radius): min(width, x + patch_radius + 1),
        ]
        valid = patch[patch > min_disparity]
        if not len(valid):
            continue

        observed = float(np.median(valid))
        homogeneous = maps["Q"] @ np.array([u, v, observed, 1.0])
        xyz_rect = homogeneous[:3] / homogeneous[3]
        xyz_world = T_world_rect[:3, :3] @ xyz_rect + T_world_rect[:3, 3]

        errors_d.append(observed - expected)
        errors_plane.append(xyz_world[2])

    return np.asarray(errors_d), np.asarray(errors_plane)


def cloud_overlap(source, target, tolerance_mm):
    """Consistencia simétrica de nubes en mm; no estima error absoluto de pose.

    La cobertura cuenta puntos con un vecino en la otra vista dentro de la
    tolerancia. Las distancias se resumen solo para esos puntos próximos.
    """
    source_dist = np.asarray(source.compute_point_cloud_distance(target))
    target_dist = np.asarray(target.compute_point_cloud_distance(source))

    covered = np.concatenate((
        source_dist[source_dist <= tolerance_mm],
        target_dist[target_dist <= tolerance_mm],
    ))

    return {
        "overlap_fraction": float((
            np.mean(source_dist <= tolerance_mm)
            + np.mean(target_dist <= tolerance_mm)
        ) / 2),
        "matched_median_mm": float(np.median(covered)) if len(covered) else float("nan"),
        "matched_p95_mm": float(np.percentile(covered, 95)) if len(covered) else float("nan"),
    }
