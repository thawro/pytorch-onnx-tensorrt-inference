import glob
import logging

import numpy as np
from torch import nn
from torchvision.models.resnet import (
    _IMAGENET_CATEGORIES,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet50,
    resnet152,
)

from model import (
    BaseInferenceModel,
    ONNXInferenceModel,
    PytorchInferenceModel,
    TRTInferenceModel,
)
from model.config import ModelConfig
from utils import load_image, measure_time

IMAGES_FILEPATHS = glob.glob("examples/*")

DEVICE = "cuda"
# DEVICE = "cpu"

if DEVICE == "cpu":
    ONNX_PROVIDERS = ["CPUExecutionProvider"]
else:
    ONNX_PROVIDERS = ["CUDAExecutionProvider"]


@measure_time(logging.INFO, time_unit="ms")
def run_inference_n_times(model: BaseInferenceModel, image: np.ndarray, n: int = 1000):
    vals = []
    for i in range(n):
        probs = model.inference(image)
        max_idx = np.argmax(probs)
        max_val = probs[max_idx]
        vals.append((max_idx, max_val))
    return vals


def test_pytorch_model(
    module: nn.Module, image: np.ndarray, model_cfg: ModelConfig, n: int = 100
):
    logging.info(" PyTorch ".center(120, "="))
    model = PytorchInferenceModel(model_cfg, device=DEVICE)
    model.load_module(module)
    model.warmup()
    vals = run_inference_n_times(model, image, n=n)
    return vals


def test_onnx_model(
    module: nn.Module, image: np.ndarray, model_cfg: ModelConfig, n: int = 100
):
    logging.info(" ONNX ".center(120, "="))
    pytorch_model = PytorchInferenceModel(model_cfg, device=DEVICE)
    pytorch_model.load_module(module)
    pytorch_model.save_to_onnx()

    model = ONNXInferenceModel(model_cfg, providers=ONNX_PROVIDERS)
    model.load_session_from_onnx()
    model.warmup()
    vals = run_inference_n_times(model, image, n=n)
    return vals


def test_trt_model(
    module: nn.Module, image: np.ndarray, model_cfg: ModelConfig, n: int = 100
):
    logging.info(" TensorRT ".center(120, "="))
    if DEVICE != "cuda":
        e = ValueError("TensorRT doesn't support CPU device. Returning None")
        logging.exception(e)
        return
    pytorch_model = PytorchInferenceModel(model_cfg, device=DEVICE)
    pytorch_model.load_module(module)
    pytorch_model.save_to_onnx()

    model = TRTInferenceModel(model_cfg)
    model.load_engine_from_onnx()
    # try:
    #     model.load_engine_from_trt()
    # except FileNotFoundError as e:
    #     logging.warning(e)
    #     model.load_engine_from_onnx()
    #     model.save_engine_to_trt()
    input_cfg = model_cfg.inputs[0]
    h, w, c = input_cfg.shapes.example
    context = model.create_context()
    stream = model.create_stream()
    context.set_optimization_profile_async(0, stream)
    context.set_input_shape(input_cfg.name, (1, c, h, w))
    model.allocate_buffers()
    model.warmup()
    vals = run_inference_n_times(model, image, n=n)
    model.free_buffers()
    return vals


if __name__ == "__main__":
    model_cfg = ModelConfig.from_yaml("model_config.yaml")

    image_filepath = IMAGES_FILEPATHS[0]
    image = load_image(image_filepath)
    n = 10

    module = resnet152(weights=ResNet152_Weights.DEFAULT)
    module.eval()

    params = dict(module=module, n=n, image=image, model_cfg=model_cfg)

    pytorch_vals = test_pytorch_model(**params)
    onnx_vals = test_onnx_model(**params)
    trt_vals = test_trt_model(**params)

    logging.info(" Returned values ".center(120, "="))
    print(f"PyTorch: {pytorch_vals}")
    print(f"ONNX: {onnx_vals}")
    print(f"TensorRT: {trt_vals}")
