import logging
from pathlib import Path

from torch import nn

from run_measurements import RESULTS_DIR
from src.engines import PyTorchInferenceEngine, TensorRTInferenceEngine
from src.engines.config import EngineConfig
from src.utils.args import parse_args
from src.utils.load import load_engine_cfg, load_pytorch_module


def save_onnx_engine(module: nn.Module, engine_cfg: EngineConfig):
    logging.info(" ONNX ".center(120, "="))

    pytorch_engine = PyTorchInferenceEngine(engine_cfg, device="cpu")
    pytorch_engine.load_module(module, device="cpu")
    pytorch_engine.save_to_onnx(model_dirpath)
    pytorch_engine.free_buffers()
    del pytorch_engine
    logging.info("Saved ONNX Engine")


def save_trt_engine(module: nn.Module, engine_cfg: EngineConfig):
    logging.info(" TensorRT ".center(120, "="))

    pytorch_engine = PyTorchInferenceEngine(engine_cfg, device="cpu")
    pytorch_engine.load_module(module, device="cpu")
    pytorch_engine.save_to_onnx(model_dirpath)
    pytorch_engine.free_buffers()
    del pytorch_engine

    trt_engine = TensorRTInferenceEngine(engine_cfg)
    trt_engine.load_engine_from_onnx(model_dirpath)
    trt_engine.save_engine_to_trt(model_dirpath)
    logging.info("Saved TensorRT Engine")


if __name__ == "__main__":
    args = parse_args()
    logging.info(f"-> Preparing engines for arguments: \n{args}")
    model_name = args.model_name
    engine_cfg = load_engine_cfg(model_name)
    pytorch_module = load_pytorch_module(model_name)

    model_dirpath = f"{RESULTS_DIR}/{engine_cfg.name}"
    Path(model_dirpath).mkdir(exist_ok=True, parents=True)
    engine_cfg.save_to_yaml(f"{model_dirpath}/config.yaml")

    save_onnx_engine(module=pytorch_module, engine_cfg=engine_cfg)
    save_trt_engine(module=pytorch_module, engine_cfg=engine_cfg)

    logging.info(" Saved Engines ".center(120, "="))
