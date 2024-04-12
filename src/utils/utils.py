import numpy as np
import yaml
from PIL import Image


def load_image(filepath: str) -> np.ndarray:
    return np.asarray(Image.open(filepath).convert("RGB"))


def save_yaml(dct: dict, filepath: str):
    with open(filepath, "w") as outfile:
        yaml.dump(dct, outfile)


def load_yaml(filepath: str) -> dict:
    with open(filepath) as stream:
        obj = yaml.safe_load(stream)
    return obj
