import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.pose import world_from_rectified


def plot_epipolar_pair(left, right, n_lines: int = 8) -> None:
    image = np.hstack((left, right))
    height = left.shape[0]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for y in np.linspace(0, height - 1, n_lines + 2)[1:-1]:
        ax.axhline(y, color='yellow', linewidth=0.8, alpha=0.7)

    ax.axis('off')
    plt.show()


def plot_disparity(disparity: np.ndarray, min_disparity: int) -> None:
    valid = disparity > min_disparity
    masked = np.ma.masked_where(~valid, disparity)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(masked, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Disparidad (px)")
    ax.set_title("Disparidad estéreo")
    ax.axis("off")
    plt.show()


def plot_pointcloud(points: np.ndarray, colors: np.ndarray, max_points: int) -> None:
    step = max(1, int(np.ceil(len(points) / max_points)))
    sampled_points = points[::step]
    # Matplotlib espera colores RGB normalizados entre 0 y 1.
    sampled_colors = colors[::step] / 255.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*sampled_points.T, c=sampled_colors, s=0.4)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Nube 3D local")
    plt.show()


def plot_pose(image, pose, K, dist, axis_length_mm, T_world_board) -> None:
    result = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(result, pose.corners, pose.ids)
    R, _ = cv2.Rodrigues(pose.rvec)
    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = R
    T_camera_board[:3, 3] = pose.tvec.reshape(3)
    # PnP da tablero -> cámara, componemos la pose de mundo -> cámara.
    T_camera_world = T_camera_board @ np.linalg.inv(T_world_board)
    rvec_world, _ = cv2.Rodrigues(T_camera_world[:3, :3])
    cv2.drawFrameAxes(
        result, K, dist, rvec_world, T_camera_world[:3, 3], axis_length_mm
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Pose del tablero (RMSE de la pose: {pose.rmse:.2f} px)")
    ax.axis("off")
    plt.show()


def show_world_cloud(cloud, camera_transforms, axis_size_mm: float) -> None:
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size_mm)
    camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size_mm)
    # Se muestran los ejes de la primera cámara y la trayectoria completa.
    camera_axes.transform(camera_transforms[0])

    centers = [T[:3, 3] for T in camera_transforms]
    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(centers)
    trajectory.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(centers) - 1)]
    )
    trajectory.paint_uniform_color((1.0, 0.2, 0.1))

    o3d.visualization.draw_geometries([cloud, world_axes, camera_axes, trajectory])


def plot_camera_poses(
    poses,
    selected_ids,
    R1,
    T_world_board,
    K_rect,
    image_size,
    board_size_mm,
    cloud=None,
    frustum_depth_mm=90.0,
):
    """Dibuja cámaras, tablero y nube en el marco global Z positivo."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    width, height = board_size_mm
    board = np.array([[0, 0, 0], [width, 0, 0],
                      [width, height, 0], [0, height, 0], [0, 0, 0]])
    board = board @ T_world_board[:3, :3].T + T_world_board[:3, 3]
    ax.plot(*board.T, color="black", linewidth=1.5)

    for axis, color in zip(np.eye(3) * 84, ("red", "green", "blue")):
        ax.plot(*np.array([[0, 0, 0], axis]).T, color=color, linewidth=2)

    if cloud is not None:
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        step = max(1, int(np.ceil(len(points) / 12000)))
        ax.scatter(*points[::step].T, c=colors[::step], s=0.35, alpha=0.75)

    cameras = {
        i: world_from_rectified(p, R1, T_world_board) for i, p in poses.items()
    }
    all_centers = np.array([T[:3, 3] for T in cameras.values()])
    ax.scatter(*all_centers.T, c="gray", s=11, alpha=0.5)

    # Las 4 esquinas del sensor definen el frustum a profundidad visual fija.
    w, h = image_size
    pixels = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    camera_corners = (np.linalg.solve(K_rect, pixels.T).T * frustum_depth_mm)

    centers = []
    for i in selected_ids:
        T = cameras[i]
        center = T[:3, 3]
        corners = camera_corners @ T[:3, :3].T + center
        for corner in corners:
            ax.plot(*np.array([center, corner]).T, color="darkorange", alpha=0.55, linewidth=0.8)
        ax.plot(*np.vstack((corners, corners[0])).T, color="darkorange", alpha=0.65, linewidth=0.8)
        ax.text(*center, str(i), fontsize=7)
        centers.append(center)

    centers = np.asarray(centers)
    ax.plot(*centers.T, color="darkorange", linewidth=1.5)
    ax.scatter(*centers.T, c="darkorange", s=23)

    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
           title="Poses de cámara en el marco global (Z hacia arriba)")
    scene = np.vstack((board, all_centers, centers))
    ax.set_box_aspect(np.ptp(scene, axis=0))
    return fig



def camera_poses_mesh(
    poses,
    selected_ids,
    R1,
    T_world_board,
    K_rect,
    image_size,
    board_size_mm,
    frustum_depth_mm,
    line_radius_mm,
    axis_length_mm,
):
    """Cámaras rectificadas y tablero como malla 3D exportable a PLY."""
    mesh = o3d.geometry.TriangleMesh()

    # Cada arista se representa con un cilindro orientado entre sus extremos.
    def segment(a, b, color):
        a, b = np.asarray(a, float), np.asarray(b, float)
        vector = b - a
        length = np.linalg.norm(vector)
        direction = vector / length
        reference = np.eye(3)[np.argmin(np.abs(direction))]
        x = np.cross(reference, direction)
        x /= np.linalg.norm(x)
        rotation = np.column_stack((x, np.cross(direction, x), direction))

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=line_radius_mm, height=length, resolution=6
        )
        cylinder.rotate(rotation, center=(0, 0, 0))
        cylinder.translate((a + b) / 2)
        cylinder.paint_uniform_color(color)
        return cylinder

    board_w, board_h = board_size_mm
    board = np.array([
        [0, 0, 0], [board_w, 0, 0],
        [board_w, board_h, 0], [0, board_h, 0],
    ], dtype=float)
    board = board @ T_world_board[:3, :3].T + T_world_board[:3, 3]
    for i in range(4):
        mesh += segment(board[i], board[(i + 1) % 4], (0.35, 0.35, 0.35))

    origin = np.zeros(3)
    for direction, color in zip(np.eye(3) * axis_length_mm,
                                ((1, 0, 0), (0, 0.8, 0), (0, 0.3, 1))):
        mesh += segment(origin, direction, color)

    width, height = image_size
    pixels = np.array([
        [0, 0, 1], [width, 0, 1],
        [width, height, 1], [0, height, 1],
    ], dtype=float)
    local_corners = np.linalg.solve(K_rect, pixels.T).T * frustum_depth_mm
    orange = (1.0, 0.45, 0.05)

    for i in selected_ids:
        T = world_from_rectified(poses[i], R1, T_world_board)
        center = T[:3, 3]
        corners = local_corners @ T[:3, :3].T + center
        for corner in corners:
            mesh += segment(center, corner, orange)
        for j in range(4):
            mesh += segment(corners[j], corners[(j + 1) % 4], orange)

    mesh.compute_vertex_normals()
    return mesh


def plot_measurement(cloud, bbox):
    """Muestra la caja orientada en XY, con altura respecto del plano ChArUco."""
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    step = max(1, int(np.ceil(len(points) / 40000)))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        *points[::step].T, c=colors[::step], s=0.8,
        alpha=1.0, depthshade=False, linewidths=0,
    )

    # Vértices y 12 aristas de la caja de medición orientados
    signs = np.array([[x, y, z] for x in (-0.5, 0.5)
                      for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    corners = (signs * bbox.extent) @ bbox.R.T + bbox.center
    # El XOR cambia un eje por vez y evita repetir aristas.
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i:
                ax.plot(*corners[[i, j]].T, color="darkorange", linewidth=1.5)

    # Aristas de referencia para identificar largo, ancho y altura.
    origin = bbox.center - bbox.R @ (bbox.extent / 2)
    for i, (name, color) in enumerate(
        (("Largo", "red"), ("Ancho", "green"), ("Alto", "blue"))
    ):
        end = origin + bbox.extent[i] * bbox.R[:, i]
        ax.plot(*np.vstack((origin, end)).T, color=color, linewidth=2)
        ax.text(*end, f"{name}: {bbox.extent[i]:.1f} mm", color=color)

    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
           title="Medición en marco global (Z hacia arriba)")
    ax.set_box_aspect(np.ptp(np.vstack((points, corners)), axis=0))
    return fig