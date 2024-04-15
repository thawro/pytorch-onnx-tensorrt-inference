import numpy as np
from torch import nn

from src.engines.config import EngineConfig
from src.loaders import *  # noqa
from src.utils.utils import MODEL_LOADER_REGISTRY


def load_engine_cfg(model_name: str) -> EngineConfig:
    engine_loader = MODEL_LOADER_REGISTRY[model_name]
    engine_cfg = engine_loader.load_engine_cfg()
    return engine_cfg


def load_pytorch_module(model_name: str) -> nn.Module:
    engine_loader = MODEL_LOADER_REGISTRY[model_name]
    pytorch_module = engine_loader.load_pytorch_module()
    return pytorch_module


def load_example_inputs(model_name: str) -> list[np.ndarray]:
    engine_loader = MODEL_LOADER_REGISTRY[model_name]
    example_inputs = engine_loader.load_example_inputs()
    return example_inputs
