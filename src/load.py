from typing import Literal

from torch import nn

import models.resnet152.load as model_resnet_152
from src.engines.config import EngineConfig

_model_name = Literal["resnet152"]


model2module = {
    "resnet152": model_resnet_152,
}


def load_engine_cfg(model_name: _model_name) -> EngineConfig:
    python_module = model2module[model_name]
    engine_cfg = python_module.load_engine_cfg()
    return engine_cfg


def load_pytorch_module(model_name: _model_name) -> nn.Module:
    python_module = model2module[model_name]
    pytorch_module = python_module.load_pytorch_module()
    return pytorch_module
