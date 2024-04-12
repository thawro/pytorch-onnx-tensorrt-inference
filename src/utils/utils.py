import numpy as np
from PIL import Image


def load_image(filepath: str) -> np.ndarray:
    return np.asarray(Image.open(filepath).convert("RGB"))
