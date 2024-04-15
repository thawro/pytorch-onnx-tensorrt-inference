import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch import nn

from src.engines.config import EngineConfig
from src.utils.utils import load_example_image_inputs


class BaseEngineLoader(ABC):
    name: str

    def _get_engine_cfg_filepath(self) -> str:
        module_filepath = inspect.getfile(self.__class__)
        cfg_dirpath = Path(module_filepath).parent.absolute()
        cfg_filepath = cfg_dirpath / "config.yaml"
        return str(cfg_filepath)

    def load_engine_cfg(self) -> EngineConfig:
        cfg_filepath = self._get_engine_cfg_filepath()
        logging.info(f"-> Loading config from {cfg_filepath}")
        return EngineConfig.from_yaml(cfg_filepath)

    @abstractmethod
    def load_example_inputs(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def load_pytorch_module(self) -> nn.Module:
        raise NotImplementedError()


class ImageModelEngineLoader(BaseEngineLoader):
    def load_example_inputs(self) -> list[np.ndarray]:
        return load_example_image_inputs()
