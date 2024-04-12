import glob
import logging
import os
import time
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
from src.monitoring.system import SystemMetricsMonitor
from src.utils import load_image, measure_time
from src.utils.visualization import plot_measurements

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
    engine: BaseInferenceEngine,
    image: np.ndarray,
    system_monitor: SystemMetricsMonitor,
    n: int = 1000,
) -> list[tuple]:
    @measure_time(time_unit="ms")
    def _run_inference_n_times() -> list[tuple]:
        outputs = []
        for i in range(n):
            probs = engine.inference(image)
            label_idx = np.argmax(probs)
            label = _IMAGENET_CATEGORIES[label_idx]
            label_prob = probs[label_idx]
            outputs.append((label_idx, label, round(label_prob, 2)))
        return outputs

    time.sleep(1)
    system_monitor.start()
    engine.warmup(WARMUP)
    outputs = _run_inference_n_times()
    system_monitor.finish()
    engine.free_buffers()
    time.sleep(1)
    return outputs


def test_pytorch_model(
    module: nn.Module,
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" PyTorch ".center(120, "="))
    model = PyTorchInferenceEngine(engine_cfg, device=device)
    system_monitor = SystemMetricsMonitor(name=model.name)
    model.load_module(module, device=device)
    system_monitor.checkpoint_metrics("loaded_engine")
    outputs = run_inference_n_times(model, image, system_monitor, n=n)
    return outputs


def test_onnx_model(
    module: nn.Module,
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" ONNX ".center(120, "="))
    providers = (
        ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    )

    onnx_engine = ONNXInferenceEngine(engine_cfg, providers=providers)

    if os.path.exists(f"{model_dirpath}/{engine_cfg.onnx_filename}"):
        system_monitor = SystemMetricsMonitor(name=onnx_engine.name)
        onnx_engine.load_session_from_onnx(model_dirpath)
    else:
        pytorch_engine = PyTorchInferenceEngine(engine_cfg, device=device)
        pytorch_engine.load_module(module, device="cpu")
        pytorch_engine.save_to_onnx(model_dirpath)
        pytorch_engine.free_buffers()
        del pytorch_engine

        system_monitor = SystemMetricsMonitor(name=onnx_engine.name)
        onnx_engine.load_session_from_onnx(model_dirpath)

    system_monitor.checkpoint_metrics("loaded_engine")
    onnx_engine.allocate_buffers()
    system_monitor.checkpoint_metrics("allocated_buffers")
    outputs = run_inference_n_times(onnx_engine, image, system_monitor, n=n)
    return outputs


def test_trt_model(
    module: nn.Module,
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
    n: int = 100,
) -> list[tuple]:
    logging.info(" TensorRT ".center(120, "="))
    if device != "cuda":
        e = ValueError("TensorRT doesn't support CPU device. Returning None")
        logging.exception(e)
        return []

    trt_engine = TensorRTInferenceEngine(engine_cfg)
    if os.path.exists(f"{model_dirpath}/{engine_cfg.trt_filename}"):
        system_monitor = SystemMetricsMonitor(name=trt_engine.name)
        trt_engine.load_engine_from_trt(model_dirpath)
    else:
        pytorch_engine = PyTorchInferenceEngine(engine_cfg, device=device)
        pytorch_engine.load_module(module, device="cpu")
        pytorch_engine.save_to_onnx(model_dirpath)
        pytorch_engine.free_buffers()
        del pytorch_engine
        system_monitor = SystemMetricsMonitor(name=trt_engine.name)
        trt_engine.load_engine_from_onnx(model_dirpath)
        trt_engine.save_engine_to_trt(model_dirpath)

    system_monitor.checkpoint_metrics("loaded_engine")
    input_cfg = engine_cfg.inputs[0]
    h, w, c = input_cfg.shapes.example
    context = trt_engine.create_context()
    stream = trt_engine.create_stream()
    context.set_optimization_profile_async(0, stream)
    context.set_input_shape(input_cfg.name, (1, c, h, w))
    trt_engine.allocate_buffers()
    system_monitor.checkpoint_metrics("allocated_buffers")
    outputs = run_inference_n_times(trt_engine, image, system_monitor, n=n)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    device = args.device

    engine_cfg = EngineConfig.from_yaml("model_config.yaml")

    model_dirpath = f"{MODELS_DIR}/{engine_cfg.name}"
    Path(model_dirpath).mkdir(exist_ok=True, parents=True)
    engine_cfg.save_to_yaml(f"{model_dirpath}/config.yaml")

    image_filepath = IMAGES_FILEPATHS[0]
    image = load_image(image_filepath)

    module = resnet152(weights=ResNet152_Weights.DEFAULT)
    module.fc = nn.Sequential(*[module.fc, nn.Softmax()])  # logits -> probs
    module.eval()

    params = dict(module=module, n=N, image=image, engine_cfg=engine_cfg, device=device)

    trt_outputs = test_trt_model(**params)
    onnx_outputs = test_onnx_model(**params)
    pytorch_outputs = test_pytorch_model(**params)

    logging.info(
        " Parsed engines outputs ([<label_idx>, <label>, <label_prob>]) ".center(
            120, "="
        )
    )
    # print(f"PyTorch: {pytorch_outputs}\n")
    # print(f"ONNX: {onnx_outputs}\n")
    # print(f"TensorRT: {trt_outputs}\n")
    plot_measurements(
        names=[
            PyTorchInferenceEngine.name,
            ONNXInferenceEngine.name,
            TensorRTInferenceEngine.name,
        ],
        time_unit="ms",
        skip_first_n=WARMUP,
        device=device,
        dirpath=f"models/{engine_cfg.name}",
    )
