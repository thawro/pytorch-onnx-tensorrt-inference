from run_measurements import MODELS_DIR, NUM_INFERENCE_ITER, NUM_WARMUP_ITER
from src.engines import (
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.load import load_engine_cfg
from src.utils import load_yaml
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
        dct = load_yaml(f"{model_dirpath}/results/cuda_{name}_latency.yaml")
        cuda_time_measurements.update(dct)

    gpu_memory_measurements = {}
    for name in names:
        dct = load_yaml(f"{model_dirpath}/results/{name}_gpu_memory.yaml")
        gpu_memory_measurements.update(dct)

    cpu_time_measurements = {}
    for name in names:
        if name == "TensorRT":
            continue
        dct = load_yaml(f"{model_dirpath}/results/cpu_{name}_latency.yaml")
        cpu_time_measurements.update(dct)
    print(cpu_time_measurements)

    plot_measurements(
        f"{model_dirpath}/plots",
        NUM_WARMUP_ITER,
        cuda_time_measurements,
        cpu_time_measurements,
        gpu_memory_measurements,
    )
