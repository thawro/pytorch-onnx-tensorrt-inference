import glob
import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from src.engines import (
    BaseInferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.engines.config import EngineConfig
from src.load import _IMAGENET_CATEGORIES, load_engine_cfg, load_pytorch_module
from src.monitoring.system import MEMORY_MEASUREMENTS, SystemMetricsMonitor
from src.monitoring.time import TIME_MEASUREMENTS
from src.utils import load_image, measure_time, save_yaml
from src.utils.visualization import plot_measurements

IMAGES_FILEPATHS = glob.glob("examples/*")

NUM_WARMUP_ITER = 100
NUM_INFERENCE_ITER = 1000

MODELS_DIR = "models"


def parse_args() -> Namespace:
    parser = ArgumentParser("PyTorch vs ONNX vs TensorRT inference comparison")
    parser.add_argument(
        "--device",
        type=str,
        help="Inference device ('cpu' or 'cuda'). cuda available for all engines, cpu not available for TensorRT.",
        default="cuda",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Inference engine ('trt', 'onnx' or 'pytorch').",
        default="trt",
    )
    return parser.parse_args()


def run_inference_n_times(
    engine: BaseInferenceEngine,
    image: np.ndarray,
    system_monitor: SystemMetricsMonitor,
) -> list[tuple]:
    @measure_time(time_unit="ms")
    def _run_inference_n_times() -> list[tuple]:
        outputs = []
        for i in range(NUM_INFERENCE_ITER):
            probs = engine.inference(image)
            label_idx = np.argmax(probs)
            label = _IMAGENET_CATEGORIES[label_idx]
            label_prob = probs[label_idx]
            outputs.append((label_idx, label, round(label_prob, 2)))
        return outputs

    time.sleep(1)
    system_monitor.start()
    engine.warmup(NUM_WARMUP_ITER)
    outputs = _run_inference_n_times()
    system_monitor.finish()
    engine.free_buffers()
    time.sleep(1)
    return outputs


def test_pytorch_engine(
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
) -> list[tuple]:
    logging.info(" PyTorch ".center(120, "="))
    module = load_pytorch_module()
    logging.info("Loaded PyTorch Module")
    model = PyTorchInferenceEngine(engine_cfg, device=device)
    system_monitor = SystemMetricsMonitor(name=model.name)
    model.load_module(module, device=device)
    system_monitor.checkpoint_metrics("loaded_engine")
    outputs = run_inference_n_times(model, image, system_monitor)
    return outputs


def test_onnx_engine(
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
) -> list[tuple]:
    logging.info(" ONNX ".center(120, "="))
    providers = (
        ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    )
    onnx_engine = ONNXInferenceEngine(engine_cfg, providers=providers)
    system_monitor = SystemMetricsMonitor(name=onnx_engine.name)
    onnx_engine.load_session_from_onnx(model_dirpath)
    logging.info("Loaded Session from ONNX file")
    system_monitor.checkpoint_metrics("loaded_engine")
    onnx_engine.allocate_buffers()
    system_monitor.checkpoint_metrics("allocated_buffers")
    outputs = run_inference_n_times(onnx_engine, image, system_monitor)
    return outputs


def test_trt_engine(
    image: np.ndarray,
    engine_cfg: EngineConfig,
    device: str,
) -> list[tuple]:
    logging.info(" TensorRT ".center(120, "="))
    if device != "cuda":
        e = ValueError("TensorRT doesn't support CPU device. Returning None")
        logging.exception(e)
        return []

    trt_engine = TensorRTInferenceEngine(engine_cfg)
    system_monitor = SystemMetricsMonitor(name=trt_engine.name)
    trt_engine.load_engine_from_trt(model_dirpath)
    logging.info("Loaded Engine from TRT file")
    system_monitor.checkpoint_metrics("loaded_engine")
    input_cfg = engine_cfg.inputs[0]
    h, w, c = input_cfg.shapes.example
    context = trt_engine.create_context()
    stream = trt_engine.create_stream()
    context.set_optimization_profile_async(0, stream)
    context.set_input_shape(input_cfg.name, (1, c, h, w))
    trt_engine.allocate_buffers()
    system_monitor.checkpoint_metrics("allocated_buffers")
    outputs = run_inference_n_times(trt_engine, image, system_monitor)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    engine = args.engine

    engine_cfg = load_engine_cfg()

    model_dirpath = f"{MODELS_DIR}/{engine_cfg.name}"
    Path(model_dirpath).mkdir(exist_ok=True, parents=True)
    engine_cfg.save_to_yaml(f"{model_dirpath}/config.yaml")

    image_filepath = IMAGES_FILEPATHS[0]
    image = load_image(image_filepath)

    module = load_pytorch_module()

    engine_fns = {
        "TensorRT": test_trt_engine,
        "ONNX": test_onnx_engine,
        "PyTorch": test_pytorch_engine,
    }
    assert (
        engine in engine_fns.keys()
    ), f"Engine must be one of {list(engine_fns.keys())}"

    test_engine_fn = engine_fns[engine]

    outputs = test_engine_fn(image=image, engine_cfg=engine_cfg, device=device)

    logging.info(" Finished inference ".center(120, "="))

    time_measurements = dict(TIME_MEASUREMENTS["ms"])
    time_measurements = {k: v[NUM_WARMUP_ITER:] for k, v in time_measurements.items()}
    save_yaml(time_measurements, f"{model_dirpath}/{device}_{engine}_latency.yaml")
