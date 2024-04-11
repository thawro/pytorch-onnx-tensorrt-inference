"""Contains functions to measure time of wraped methods, functions."""

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)


def load_image(filepath: str) -> np.ndarray:
    return np.asarray(Image.open(filepath).convert("RGB"))


def parse_time(time_in_sec: float, target_unit: str = "ms") -> str:
    sec2unit = {
        "h": time_in_sec / 3600,
        "m": time_in_sec / 60,
        "s": time_in_sec,
        "ms": time_in_sec * 1e3,
        "us": time_in_sec * 1e6,
        "ns": time_in_sec * 1e9,
        "ps": time_in_sec * 1e12,
    }
    return f"{sec2unit[target_unit]:.3f} [{target_unit}]"


def measure_time(log_level: int = logging.DEBUG, time_unit: str = "ms") -> Callable:
    """Measure function execution time"""

    def function_measure_time(function: Callable) -> Callable:
        """Measure function execution time."""

        @wraps(function)
        def wrapper_function_measure_time(*args: Any, **kwargs: Any) -> Any:
            func_filename = Path(function.__code__.co_filename)

            start = time.time()
            return_val = function(*args, **kwargs)
            end = time.time()
            duration_s = end - start
            msg = (
                f"{function.__name__} "
                f"Execution time: {parse_time(duration_s, time_unit)}"
            )

            if log_level == logging.INFO:
                logging.info(msg)

            elif log_level == logging.DEBUG:
                msg = msg + f"\n{func_filename}"
                logging.debug(msg)

            return return_val

        return wrapper_function_measure_time

    return function_measure_time
