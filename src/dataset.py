from pathlib import Path

import cv2


def load_stereo_pairs(directory: Path) -> dict[int, tuple[Path, Path]]:
    left = {int(p.stem.split('_')[-1]): p for p in directory.glob('left_*.jpg')}
    right = {int(p.stem.split('_')[-1]): p for p in directory.glob('right_*.jpg')}

    return {i: (left[i], right[i]) for i in sorted(left)}


def read_stereo_pair(paths: tuple[Path, Path]):
    left_path, right_path = paths
    return cv2.imread(str(left_path)), cv2.imread(str(right_path))
