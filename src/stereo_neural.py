from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class CREStereo:
    def __init__(self, model_path: Path):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        height, width = self.session.get_inputs()[-1].shape[-2:]
        self.size = (width, height)

    def _tensor(self, image: np.ndarray, half: bool = False) -> np.ndarray:
        width, height = self.size
        if half:
            width, height = width // 2, height // 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image.transpose(2, 0, 1)[None].astype(np.float32)

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        inputs = [
            self._tensor(left, half=True),
            self._tensor(right, half=True),
            self._tensor(left),
            self._tensor(right),
        ]
        output = self.session.run(None, dict(zip(self.input_names, inputs)))[0]
        return output[0, 0].astype(np.float32)
