import numpy as np
import open3d as o3d


def make_pointcloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return cloud


def world_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    T_world_rect: np.ndarray,
    roi: o3d.geometry.AxisAlignedBoundingBox,
) -> o3d.geometry.PointCloud:
    """Transforma puntos rectificados al marco ChArUco y recorta antes de crear la nube."""
    world_points = points @ T_world_rect[:3, :3].T + T_world_rect[:3, 3]
    inside = np.all(
        (world_points >= roi.get_min_bound()) & (world_points <= roi.get_max_bound()),
        axis=1,
    )
    return make_pointcloud(world_points[inside], colors[inside])


def measure_object(
    cloud: o3d.geometry.PointCloud,
    trim_percent: float = 1.0,
) -> tuple[dict[str, float], o3d.geometry.OrientedBoundingBox]:
    """Largo y ancho por PCA en XY, altura sobre el tablero (Z >= 0)."""
    points = np.asarray(cloud.points)
    xy = points[:, :2]
    _, axes = np.linalg.eigh(np.cov(xy.T))
    major = axes[:, -1]
    minor = np.array([-major[1], major[0]])

    rotation = np.eye(3)
    rotation[:2, 0] = major
    rotation[:2, 1] = minor
    rotation[2, 2] = 1.0

    coordinates = points @ rotation
    low, high = np.percentile(
        coordinates, [trim_percent, 100 - trim_percent], axis=0
    )
    low[2] = 0.0  # Plano de apoyo conocido, la segmentación recorta la suela.

    extent = high - low
    center = rotation @ ((low + high) / 2)
    bbox = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
    bbox.color = (1.0, 0.65, 0.1)

    measurements = {
        "length_mm": float(extent[0]),
        "width_mm": float(extent[1]),
        "height_mm": float(extent[2]),
    }
    return measurements, bbox
