from torch import nn
from torchvision.models.resnet import (
    _IMAGENET_CATEGORIES,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet50,
    resnet152,
)

from src.engines.config import EngineConfig


def load_pytorch_module() -> nn.Module:
    module = resnet152(weights=ResNet152_Weights.DEFAULT)
    module.fc = nn.Sequential(*[module.fc, nn.Softmax()])  # logits -> probs
    module.eval()
    return module


def load_engine_cfg() -> EngineConfig:
    engine_cfg = EngineConfig.from_yaml("model_config.yaml")
    return engine_cfg
