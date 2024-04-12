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


def defaultdict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = defaultdict2dict(v)
    return dict(d)


def subtract_init_memory_usage(
    memory_measurements: dict, ckpt_memory_measurements: dict
) -> dict:
    _memory_measurements = memory_measurements.copy()
    for engine_name, memory_ckpt_stats in ckpt_memory_measurements.items():
        init_stats = memory_ckpt_stats["initialized"]
        # NOTE: remove init value of memory (mb) stats to know how much increased
        for stat_name, init_value in init_stats.items():
            if "mb" in stat_name:
                values = _memory_measurements[engine_name][stat_name]
                values = [value - init_value for value in values]
                _memory_measurements[engine_name][stat_name] = values
    return _memory_measurements
