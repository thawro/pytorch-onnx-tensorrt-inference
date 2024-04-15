import logging
import time
from pathlib import Path

import numpy as np

from src.engines import (
    BaseInferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.engines.config import EngineConfig
from src.load import load_engine_cfg, load_example_inputs, load_pytorch_module
from src.monitoring.system import SystemMetricsMonitor
from src.utils import measure_time, save_yaml
from src.utils.args import parse_args
from src.utils.measurements import prepare_measurements

RESULTS_DIR = "measurements_results"


def run_inference_n_times(
    engine: BaseInferenceEngine,
    example_inputs: list[np.ndarray],
    system_monitor: SystemMetricsMonitor,
) -> list[tuple]:
    @measure_time(time_unit="ms")
    def _run_inference_n_times() -> list[tuple]:
        outputs_avg = []
        for i in range(args.num_iter):
            outputs = engine.inference(example_inputs)
            outputs_avg.append([out.mean() for out in outputs])
        return outputs_avg

    time.sleep(1)
    system_monitor.start()
    engine.warmup(args.num_warmup_iter)
    out = _run_inference_n_times()
    system_monitor.finish()
    engine.free_buffers()
    time.sleep(1)
    return out


def test_pytorch_engine(
    example_inputs: list[np.ndarray],
    engine_cfg: EngineConfig,
    device: str,
) -> list[tuple]:
    logging.info(" PyTorch ".center(120, "="))
    engine = PyTorchInferenceEngine(engine_cfg, device=device)
    system_monitor = SystemMetricsMonitor(name=engine.name)
    module = load_pytorch_module(model_name)
    logging.info("Loaded PyTorch Module")
    engine.load_module(module, device=device)
    system_monitor.checkpoint_metrics("loaded_engine")
    outputs = run_inference_n_times(engine, example_inputs, system_monitor)
    return outputs


def test_onnx_engine(
    example_inputs: list[np.ndarray],
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
    outputs = run_inference_n_times(onnx_engine, example_inputs, system_monitor)
    return outputs


def test_trt_engine(
    example_inputs: list[np.ndarray],
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
    outputs = run_inference_n_times(trt_engine, example_inputs, system_monitor)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"-> Running measurements for args: \n{args}")
    device = args.device
    engine = args.engine
    model_name = args.model_name
    num_warmup_iter = args.num_warmup_iter

    engine_fns = {
        "TensorRT": test_trt_engine,
        "ONNX": test_onnx_engine,
        "PyTorch": test_pytorch_engine,
    }

    assert (
        engine in engine_fns.keys()
    ), f"Engine must be one of {list(engine_fns.keys())}"

    engine_cfg = load_engine_cfg(model_name)
    example_inputs = load_example_inputs(model_name)

    # change example inputs shapes according to CLI args
    if args.example_shapes is not None:
        for i in range(len(engine_cfg.inputs)):
            engine_cfg.inputs[i].shapes.example = args.example_shapes[i]

    model_dirpath = f"{RESULTS_DIR}/{engine_cfg.name}"
    Path(model_dirpath).mkdir(exist_ok=True, parents=True)
    engine_cfg.save_to_yaml(f"{model_dirpath}/config.yaml")

    test_engine_fn = engine_fns[engine]

    outputs = test_engine_fn(
        example_inputs=example_inputs, engine_cfg=engine_cfg, device=device
    )

    logging.info(" Finished inference ".center(120, "="))
    logging.info(f"Outputs: {outputs}")

    results_dirpath = f"{model_dirpath}/results/{engine_cfg.example_inputs_shapes_str}"
    Path(results_dirpath).mkdir(exist_ok=True, parents=True)

    time_measurements, memory_measurements = prepare_measurements(
        skip_first_n=num_warmup_iter
    )
    save_yaml(memory_measurements, f"{results_dirpath}/{device}_{engine}_memory.yaml")
    save_yaml(time_measurements, f"{results_dirpath}/{device}_{engine}_latency.yaml")
