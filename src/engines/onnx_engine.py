import logging

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
        self.device = "cpu" if "CPUExecutionProvider" in self.providers else "cuda"
        self.device_id = 0

    def allocate_buffers(self):
        if not self.use_gpu:
            return
        for inp in self.cfg.inputs:
            ort_model_input = self.preprocess(np.random.randn(*inp.shapes.example))
            self.io_binding.bind_ortvalue_input(
                name=self.cfg.inputs[0].name,
                ortvalue=ort_model_input,
            )
        for out in self.cfg.outputs:
            ort_output = ort.OrtValue.ortvalue_from_shape_and_type(
                out.shape.runtime, np.float32, self.device, self.device_id
            )
            self.io_binding.bind_ortvalue_output(name=out.name, ortvalue=ort_output)

    def load_session_from_onnx(self, dirpath: str):
        self.session = ort.InferenceSession(
            f"{dirpath}/{self.cfg.onnx_filename}",
            providers=self.providers,
        )
        if self.use_gpu:
            self.io_binding = self.session.io_binding()

    @property
    def use_gpu(self) -> bool:
        return self.device != "cpu"

    def preprocess(self, image: np.ndarray) -> ort.OrtValue | np.ndarray:
        h, w, c = self.example_input_shapes[0]
        dtype = self.dtypes[0]
        image_arr = (
            np.asarray(cv2.resize(image, (w, h))).transpose([2, 0, 1]).astype(dtype)
        )
        image_arr = np.expand_dims(image_arr, 0)
        image_arr = (image_arr / 255.0 - 0.45) / 0.225
        if self.use_gpu:
            return ort.OrtValue.ortvalue_from_numpy(
                image_arr, self.device, self.device_id
            )
        return image_arr

    def _inference_gpu(self, image: np.ndarray) -> list[np.ndarray]:
        ort_model_input = self.preprocess(image)
        self.io_binding.bind_ortvalue_input(
            name=self.cfg.inputs[0].name,
            ortvalue=ort_model_input,
        )
        self.io_binding.synchronize_inputs()
        self.session.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        outputs = self.io_binding.copy_outputs_to_cpu()
        return outputs

    def _inference_cpu(self, image: np.ndarray) -> list[np.ndarray]:
        model_input = self.preprocess(image)
        outputs = self.session.run(None, {self.cfg.inputs[0].name: model_input})
        return outputs

    @measure_time(time_unit="ms", name="ONNX")
    def inference(self, image: np.ndarray):
        if self.use_gpu:
            outputs = self._inference_gpu(image)
        else:
            outputs = self._inference_cpu(image)
        probs = outputs[0][0]
        return probs

    def free_buffers(self):
        if not self.use_gpu:
            return
        logging.info("Freeing Buffers")
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
