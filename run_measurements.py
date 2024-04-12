import glob
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from torch import nn
from torchvision.models.resnet import (
    _IMAGENET_CATEGORIES,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet50,
    resnet152,
)

from src.engines import (
    BaseInferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.engines.config import EngineConfig
from src.utils import load_image, measure_time, plot_time_measurements

IMAGES_FILEPATHS = glob.glob("examples/*")

WARMUP = 100
N = 1000

MODELS_DIR = "models"


def parse_args() -> Namespace:
    parser = ArgumentParser("PyTorch vs ONNX vs TensorRT inference comparison")
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which to perform inference ('cpu' or 'cuda'). cuda available for all engines, cpu not available for TensorRT.",
        default="cuda",
    )
    return parser.parse_args()


def run_inference_n_times(
    model: BaseInferenceEngine, image: np.ndarray, n: int = 1000
) -> list[tuple]:
    @measure_time(time_unit="ms")
    def _run_inference() -> list[tuple]:
        outputs = []
        for i in range(n):
            probs = model.inference(image)
            label_idx = np.argmax(probs)
            label = _IMAGENET_CATEGORIES[label_idx]
            label_prob = probs[label_idx]
            outputs.append((label_idx, label, round(label_prob, 2)))
        return outputs

    model.warmup(WARMUP)
    return _run_inference()


def test_pytorch_model(
    module: nn.Module,
    image: np.ndarray,
    model_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" PyTorch ".center(120, "="))
    model = PyTorchInferenceEngine(model_cfg, device=device)
    model.load_module(module)
    outputs = run_inference_n_times(model, image, n=n)
    return outputs


def test_onnx_model(
    module: nn.Module,
    image: np.ndarray,
    model_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" ONNX ".center(120, "="))
    pytorch_model = PyTorchInferenceEngine(model_cfg, device=device)
    pytorch_model.load_module(module)
    pytorch_model.save_to_onnx(model_dirpath)

    providers = (
        ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    )
    model = ONNXInferenceEngine(model_cfg, providers=providers)
    model.load_session_from_onnx(model_dirpath)
    outputs = run_inference_n_times(model, image, n=n)
    return outputs


def test_trt_model(
    module: nn.Module,
    image: np.ndarray,
    model_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" TensorRT ".center(120, "="))
    if device != "cuda":
        e = ValueError("TensorRT doesn't support CPU device. Returning None")
        logging.exception(e)
        return []
    pytorch_model = PyTorchInferenceEngine(model_cfg, device=device)
    pytorch_model.load_module(module)
    pytorch_model.save_to_onnx(model_dirpath)

    model = TensorRTInferenceEngine(model_cfg)
    model.load_engine_from_onnx(model_dirpath)
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
    outputs = run_inference_n_times(model, image, n=n)
    model.free_buffers()
    return outputs


if __name__ == "__main__":
    args = parse_args()
    device = args.device

    model_cfg = EngineConfig.from_yaml("model_config.yaml")

    model_dirpath = f"{MODELS_DIR}/{model_cfg.name}"
    Path(model_dirpath).mkdir(exist_ok=True, parents=True)
    model_cfg.save_to_yaml(f"{model_dirpath}/config.yaml")

    image_filepath = IMAGES_FILEPATHS[0]
    image = load_image(image_filepath)

    module = resnet152(weights=ResNet152_Weights.DEFAULT)
    module.fc = nn.Sequential(*[module.fc, nn.Softmax()])  # logits -> probs
    module.eval()

    params = dict(module=module, n=N, image=image, model_cfg=model_cfg, device=device)

    pytorch_outputs = test_pytorch_model(**params)
    onnx_outputs = test_onnx_model(**params)
    trt_outputs = test_trt_model(**params)

    logging.info(
        " Parsed engines outputs ([<label_idx>, <label>, <label_prob>]) ".center(
            120, "="
        )
    )
    print(f"PyTorch: {pytorch_outputs[WARMUP:WARMUP+10]}\n")
    print(f"ONNX: {onnx_outputs[WARMUP:WARMUP+10]}\n")
    print(f"TensorRT: {trt_outputs[WARMUP:WARMUP+10]}\n")
    plot_time_measurements(
        names=[
            PyTorchInferenceEngine.name,
            ONNXInferenceEngine.name,
            TensorRTInferenceEngine.name,
        ],
        time_unit="ms",
        skip_first_n=WARMUP,
        filepath=f"models/{model_cfg.name}/{device}_measurements.jpg",
    )
