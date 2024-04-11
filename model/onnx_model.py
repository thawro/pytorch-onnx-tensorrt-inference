import logging

import cv2
import numpy as np
import onnxruntime as ort

from model.config import ModelConfig
from utils import measure_time

from .base import BaseInferenceModel

DEFAULT_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]
TRT_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]


class ONNXInferenceModel(BaseInferenceModel):
    def __init__(self, cfg: ModelConfig, providers: list[str] = DEFAULT_PROVIDERS):
        super().__init__(cfg)
        self.providers = providers

    def load_session_from_onnx(self):
        self.session = ort.InferenceSession(
            self.cfg.onnx_filepath,
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

    @measure_time(logging.INFO, time_unit="ms")
    def inference(self, image: np.ndarray):
        model_input = self.preprocess(image)
        outputs = self.session.run(None, {"input": model_input})
        probs = outputs[0][0]
        return probs
