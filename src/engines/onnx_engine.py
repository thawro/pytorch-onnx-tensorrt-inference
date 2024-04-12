import cv2
import numpy as np
import onnxruntime as ort

from ..utils import measure_time
from .base import BaseInferenceEngine
from .config import EngineConfig

DEFAULT_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]
TRT_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]


class ONNXInferenceEngine(BaseInferenceEngine):
    name: str = "ONNX"

    def __init__(self, cfg: EngineConfig, providers: list[str] = DEFAULT_PROVIDERS):
        super().__init__(cfg)
        self.providers = providers

    def load_session_from_onnx(self, dirpath: str):
        self.session = ort.InferenceSession(
            f"{dirpath}/{self.cfg.onnx_filename}",
            providers=self.providers,
        )

    def preprocess(self, image: np.ndarray):
        h, w, c = self.example_input_shapes[0]
        dtype = self.dtypes[0]
        image_arr = (
            np.asarray(cv2.resize(image, (w, h))).transpose([2, 0, 1]).astype(dtype)
        )
        image_arr = np.expand_dims(image_arr, 0)
        return (image_arr / 255.0 - 0.45) / 0.225

    @measure_time(time_unit="ms", name="ONNX")
    def inference(self, image: np.ndarray):
        model_input = self.preprocess(image)
        outputs = self.session.run(None, {self.cfg.inputs[0].name: model_input})[0]
        probs = outputs[0]
        return probs
