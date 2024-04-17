import logging
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, List, Optional, Union

logging.basicConfig(level=logging.INFO)

TIME_MEASUREMENTS = defaultdict(lambda: defaultdict(list))

_time_unit = Literal["h", "m", "s", "ms", "us", "ns", "ps"]


def parse_time(time_in_sec: float, time_unit: _time_unit = "ms") -> float:
    sec2unit = {
        "h": time_in_sec / 3600,
        "m": time_in_sec / 60,
        "s": time_in_sec,
        "ms": time_in_sec * 1e3,
        "us": time_in_sec * 1e6,
        "ns": time_in_sec * 1e9,
        "ps": time_in_sec * 1e12,
    }
    return sec2unit[time_unit]


def measure_time(time_unit: _time_unit = "ms", name: Optional[str] = None) -> Callable:
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
            parsed_duration = parse_time(duration_s, time_unit)
            duration_info = f"{parsed_duration:.3f} [{time_unit}]"
            _name = f" ({name})" if name is not None else ""
            msg = f"{function.__name__}{_name} " f"Execution time: {duration_info}"
            if name is not None:
                TIME_MEASUREMENTS[time_unit][name].append(parsed_duration)

            if logging.root.level == logging.INFO:
                logging.info(msg)

            elif logging.root.level == logging.DEBUG:
                msg = msg + f"\n{func_filename}"
                logging.debug(msg)

            return return_val

        return wrapper_function_measure_time

    return function_measure_time
