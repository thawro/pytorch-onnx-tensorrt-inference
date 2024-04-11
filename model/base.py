import logging

import numpy as np

from .config import ModelConfig


class BaseInferenceModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.example_input_shapes = [inp.shapes.example for inp in cfg.inputs]
        self.dtypes = [inp.dtype for inp in cfg.inputs]

    def inference(self, *args, **kwargs):
        raise NotImplementedError()

    def _dummy_input(self, shape: list[int], dtype: np.dtype):
        return np.random.randn(*shape).astype(dtype)

    @property
    def dummy_inputs(self) -> list[np.ndarray]:
        return [
            self._dummy_input(shape, dtype)
            for shape, dtype in zip(self.example_input_shapes, self.dtypes)
        ]

    def warmup(self):
        logging.warning("WARMUP")
        for _ in range(100):
            dummy_inputs = self.dummy_inputs
            self.inference(*dummy_inputs)
