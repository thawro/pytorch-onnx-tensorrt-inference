from __future__ import annotations

import glob
import logging
from typing import TYPE_CHECKING, Type

import numpy as np
import yaml
from PIL import Image

if TYPE_CHECKING:
    from src.engines.loader import BaseEngineLoader


EXAMPLE_IMAGES_FILEPATHS = glob.glob("examples/*")
MODEL_LOADER_REGISTRY: dict[str, BaseEngineLoader] = {}


def register_model(cls: Type[BaseEngineLoader]):
    print("=" * 100, cls)
    MODEL_LOADER_REGISTRY[cls.name] = cls()
    return cls


def load_image(filepath: str) -> np.ndarray:
    return np.asarray(Image.open(filepath).convert("RGB"))


def load_example_image_inputs(
    filepath: str = EXAMPLE_IMAGES_FILEPATHS[0],
) -> list[np.ndarray]:
    logging.info(f"-> Loading image from {filepath}")
    image = load_image(filepath)
    inputs = [image]
    return inputs


def save_yaml(dct: dict, filepath: str):
    with open(filepath, "w") as outfile:
        yaml.dump(dct, outfile)


def load_yaml(filepath: str) -> dict:
    with open(filepath) as stream:
        obj = yaml.safe_load(stream)
    return obj


def defaultdict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = defaultdict2dict(v)
    return dict(d)
