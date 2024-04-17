import logging

import numpy as np
import onnxruntime as ort

from src.engines.config import EngineConfig
from src.engines.engines.base import BaseInferenceEngine
from src.monitoring.time import measure_time
from typing import List, Optional, Union


DEFAULT_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]
TRT_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]


class ONNXInferenceEngine(BaseInferenceEngine):
    name: str = "ONNX"

    def __init__(self, cfg: EngineConfig, providers: List[str] = DEFAULT_PROVIDERS):
        super().__init__(cfg)
        self.providers = providers
        self.device = "cpu" if "CPUExecutionProvider" in self.providers else "cuda"
        self.device_id = 0

    def allocate_buffers(self):
        if not self.use_gpu:
            return
        preprocessed_inputs = self.preprocess_inputs(self.dummy_inputs)
        device_inputs = self.move_inputs_to_device(preprocessed_inputs)
        for i, inp in enumerate(device_inputs):
            self.io_binding.bind_ortvalue_input(
                name=self.cfg.inputs[i].name,
                ortvalue=inp,
            )
        for out in self.cfg.outputs:
            if out.is_dynamic:
                self.io_binding.bind_output(out.name, self.device, self.device_id)
            else:
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

    def move_inputs_to_device(self, inputs: List[np.ndarray]):
        if self.use_gpu:
            return [
                ort.OrtValue.ortvalue_from_numpy(inp, self.device, self.device_id)
                for inp in inputs
            ]
        return inputs

    def _inference_gpu(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        preprocessed_inputs = self.preprocess_inputs(inputs)
        device_inputs = self.move_inputs_to_device(preprocessed_inputs)
        for i, inp in enumerate(device_inputs):
            self.io_binding.bind_ortvalue_input(
                name=self.cfg.inputs[i].name,
                ortvalue=inp,
            )
        self.io_binding.synchronize_inputs()
        self.session.run_with_iobinding(self.io_binding)
        self.io_binding.synchronize_outputs()
        outputs = self.io_binding.copy_outputs_to_cpu()
        return outputs

    def _inference_cpu(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        preprocessed_inputs = self.preprocess_inputs(inputs)
        model_input = {
            self.cfg.inputs[i].name: inp for i, inp in enumerate(preprocessed_inputs)
        }
        outputs = self.session.run(None, model_input)
        return outputs

    @measure_time(time_unit="ms", name="ONNX")
    def inference(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        if self.use_gpu:
            outputs = self._inference_gpu(inputs)
        else:
            outputs = self._inference_cpu(inputs)
        return outputs

    def free_buffers(self):
        if not self.use_gpu:
            return
        logging.info("Freeing Buffers")
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
