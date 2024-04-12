import logging
import time
from pathlib import Path

import numpy as np

from run_measurements import MODELS_DIR, NUM_INFERENCE_ITER, NUM_WARMUP_ITER
from src.engines import (
    BaseInferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.engines.config import EngineConfig
from src.load import load_engine_cfg
from src.monitoring.system import MEMORY_MEASUREMENTS, SystemMetricsMonitor
from src.monitoring.time import TIME_MEASUREMENTS
from src.utils import load_image, load_yaml, measure_time
from src.utils.visualization import plot_measurements

if __name__ == "__main__":
    engine_cfg = load_engine_cfg()
    model_dirpath = f"{MODELS_DIR}/{engine_cfg.name}"
    names = [
        PyTorchInferenceEngine.name,
        ONNXInferenceEngine.name,
        TensorRTInferenceEngine.name,
    ]
    cuda_time_measurements = {}
    for name in names:
        dct = load_yaml(f"{model_dirpath}/cuda_{name}_latency.yaml")
        cuda_time_measurements.update(dct)
    cuda_time_measurements = {
        k: v[NUM_WARMUP_ITER:] for k, v in cuda_time_measurements.items()
    }

    cpu_time_measurements = {}
    # for name in names:
    #     dct = load_yaml(f"{model_dirpath}/cpu_{name}_latency.yaml")
    #     cpu_time_measurements.update(dct)
    # print(cpu_time_measurements)
    cpu_time_measurements = {
        k: v[NUM_WARMUP_ITER:] for k, v in cpu_time_measurements.items()
    }

    plot_measurements(
        model_dirpath,
        cuda_time_measurements,
        cpu_time_measurements,
    )
