import logging

import cv2
import numpy as np

from src.engines.config import EngineConfig


class BaseInferenceEngine:
    name: str

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.example_input_shapes = [inp.shapes.example for inp in cfg.inputs]
        self.dtypes = [inp.dtype for inp in cfg.inputs]

    def inference(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

    def _dummy_input(self, shape: list[int], dtype: np.dtype):
        return np.random.randn(*shape).astype(dtype)

    @property
    def dummy_inputs(self) -> list[np.ndarray]:
        return [
            self._dummy_input(shape, dtype)
            for shape, dtype in zip(self.example_input_shapes, self.dtypes)
        ]

    def preprocess_inputs(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        image, *other = inputs
        h, w, c = self.example_input_shapes[0]
        image_arr = np.asarray(cv2.resize(image, (w, h))).transpose([2, 0, 1])
        image_arr = (image_arr / 255.0 - 0.45) / 0.225

        inputs = [image_arr, *other]
        return [
            np.expand_dims(inp, 0).astype(self.dtypes[i])
            for i, inp in enumerate(inputs)
        ]

    def move_inputs_to_device(self, inputs: list):
        raise NotImplementedError()

    def warmup(self, n: int):
        logging.warning(
            f"-> Performing {self.__class__.__name__} warmup (running inference {n} times)"
        )
        for _ in range(n):
            self.inference(self.dummy_inputs)

    def free_buffers(self):
        raise NotImplementedError()
