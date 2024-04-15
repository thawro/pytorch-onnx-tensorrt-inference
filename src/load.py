from typing import Literal

import numpy as np
from torch import nn

from models.encoder.load import EncoderEngineLoader
from models.localizer.load import LocalizerEngineLoader
from models.resnet.load import Resnet50EngineLoader, Resnet152EngineLoader
from src.engines.config import EngineConfig
from src.engines.loader import BaseEngineLoader

_model_name = Literal["resnet50", "resnet152", "localizer", "encoder"]


model2loader: dict[str, BaseEngineLoader] = {
    "resnet50": Resnet50EngineLoader(),
    "resnet152": Resnet152EngineLoader(),
    "localizer": LocalizerEngineLoader(),
    "encoder": EncoderEngineLoader(),
}


def load_engine_cfg(model_name: _model_name) -> EngineConfig:
    engine_loader = model2loader[model_name]
    engine_cfg = engine_loader.load_engine_cfg()
    return engine_cfg


def load_pytorch_module(model_name: _model_name) -> nn.Module:
    engine_loader = model2loader[model_name]
    pytorch_module = engine_loader.load_pytorch_module()
    return pytorch_module


def load_example_inputs(model_name: _model_name) -> list[np.ndarray]:
    engine_loader = model2loader[model_name]
    example_inputs = engine_loader.load_example_inputs()
    return example_inputs
