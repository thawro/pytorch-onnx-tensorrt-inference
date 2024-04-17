import logging
import os

from run_measurements import RESULTS_DIR
from src.engines import (
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    TensorRTInferenceEngine,
)
from src.utils.args import parse_args
from src.utils.load import load_engine_cfg
from src.utils.utils import load_yaml
from src.utils.visualization import plot_measurements

if __name__ == "__main__":
    args = parse_args()
    logging.info(f"-> Plotting measurements for args: \n{args}")

    model_name = args.model_name

    engine_cfg = load_engine_cfg(model_name)
    results_dirpath = f"{RESULTS_DIR}/{engine_cfg.name}/results"

    all_example_inputs_shapes = next(os.walk(results_dirpath))[1]
    names = [
        #PyTorchInferenceEngine.name,
        #ONNXInferenceEngine.name,
        TensorRTInferenceEngine.name,
    ]
    for example_inputs_shapes in all_example_inputs_shapes:
        shape_results_dirpath = f"{results_dirpath}/{example_inputs_shapes}"
        plots_dirpath = f"{RESULTS_DIR}/{engine_cfg.name}/plots/{example_inputs_shapes}"
        cpu_time_measurements = {}
        cpu_memory_measurements = {}
        cuda_time_measurements = {}
        cuda_memory_measurements = {}
        for engine_name in names:
            if engine_name != "TensorRT":
                engine_cpu_time = load_yaml(
                    f"{shape_results_dirpath}/cpu_{engine_name}_latency.yaml"
                )
                cpu_time_measurements.update(engine_cpu_time)
                engine_cpu_memory = load_yaml(
                    f"{shape_results_dirpath}/cpu_{engine_name}_memory.yaml"
                )
                cpu_memory_measurements.update(engine_cpu_memory)

            engine_cuda_time = load_yaml(
                f"{shape_results_dirpath}/cuda_{engine_name}_latency.yaml"
            )
            cuda_time_measurements.update(engine_cuda_time)
            engine_cuda_memory = load_yaml(
                f"{shape_results_dirpath}/cuda_{engine_name}_memory.yaml"
            )
            cuda_memory_measurements.update(engine_cuda_memory)

        plot_measurements(
            dirpath=plots_dirpath,
            cpu_time_measurements=cpu_time_measurements,
            cpu_memory_measurements=cpu_memory_measurements,
            cuda_time_measurements=cuda_time_measurements,
            cuda_memory_measurements=cuda_memory_measurements,
        )
