from torch import nn
from torchvision.models.resnet import (
    ResNet50_Weights,
    ResNet152_Weights,
    resnet50,
    resnet152,
)

from src.engines.config import EngineConfig
from src.engines.loader import ImageModelEngineLoader
from src.utils.utils import register_model


class BaseResnetEngineLoader(ImageModelEngineLoader):
    def load_engine_cfg(self) -> EngineConfig:
        engine_cfg = super().load_engine_cfg()
        engine_cfg.name = self.name
        return engine_cfg


@register_model
class Resnet152EngineLoader(BaseResnetEngineLoader):
    name = "resnet152"

    def load_pytorch_module(self) -> nn.Module:
        module = resnet152(weights=ResNet152_Weights.DEFAULT)
        module.fc = nn.Sequential(*[module.fc, nn.Softmax()])  # logits -> probs
        module.eval()
        return module


@register_model
class Resnet50EngineLoader(BaseResnetEngineLoader):
    name = "resnet50"

    def load_pytorch_module(self) -> nn.Module:
        module = resnet50(weights=ResNet50_Weights.DEFAULT)
        module.fc = nn.Sequential(*[module.fc, nn.Softmax()])  # logits -> probs
        module.eval()
        return module
