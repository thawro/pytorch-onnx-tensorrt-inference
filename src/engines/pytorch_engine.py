import numpy as np
import torch.onnx
from torch import nn

from ..utils import measure_time
from .base import BaseInferenceEngine
from .config import EngineConfig

DEFAULT_DEVICE = "cuda"


class PyTorchInferenceEngine(BaseInferenceEngine):
    name: str = "PyTorch"

    def __init__(self, cfg: EngineConfig, device: str = DEFAULT_DEVICE):
        super().__init__(cfg)
        self.device = device

    def load_module(self, torch_module: nn.Module, device: str | None):
        if device is None:
            device = self.device
        self.module = torch_module
        self.module.eval()
        self.module.to(device)

    def save_to_onnx(self, dirpath: str):
        dummy_inputs = self.preprocess_inputs(self.dummy_inputs)
        dummy_inputs = self.move_inputs_to_device(dummy_inputs)
        dynamic_axes = {}
        for input in self.cfg.inputs:
            input_dynamic_axes = {}
            runtime_dims = input.shapes.runtime
            dims_names = input.shapes.dims_names
            for dim_idx in range(len(runtime_dims)):
                if runtime_dims[dim_idx] == -1:
                    input_dynamic_axes[dim_idx] = dims_names[dim_idx]
            dynamic_axes[input.name] = input_dynamic_axes
        for output in self.cfg.outputs:
            output_dynamic_axes = {}
            runtime_dims = output.shape.runtime
            dims_names = output.shape.dims_names
            for dim_idx in range(len(runtime_dims)):
                if runtime_dims[dim_idx] == -1:
                    output_dynamic_axes[dim_idx] = dims_names[dim_idx]
            dynamic_axes[output.name] = output_dynamic_axes
        input_names = [input.name for input in self.cfg.inputs]
        output_names = [output.name for output in self.cfg.outputs]
        torch.onnx.export(
            self.module,
            tuple(dummy_inputs),
            f"{dirpath}/{self.cfg.onnx_filename}",
            export_params=True,  # store the trained parameter weights inside the model file
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=input_names,  # the model's input names
            output_names=output_names,  # the model's output names
            dynamic_axes=dynamic_axes,
        )
        torch.cuda.empty_cache()

    def move_inputs_to_device(self, inputs: list[np.ndarray]):
        device = next(self.module.parameters()).device
        return [torch.from_numpy(inp).to(device) for inp in inputs]

    @measure_time(time_unit="ms", name="PyTorch")
    def inference(self, inputs: list[np.ndarray]):
        with torch.no_grad():
            preprocessed_inputs = self.preprocess_inputs(inputs)
            inputs_on_device = self.move_inputs_to_device(preprocessed_inputs)
            probs = self.module(*inputs_on_device)[0].cpu().numpy()
        return probs

    def free_buffers(self):
        self.module.to("cpu")
        torch.cuda.empty_cache()
