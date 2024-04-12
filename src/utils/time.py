import logging
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)

MEASUREMENTS = defaultdict(lambda: defaultdict(list))

_unit = Literal["h", "m", "s", "ms", "us", "ns", "ps"]


def parse_time(time_in_sec: float, time_unit: _unit = "ms") -> float:
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


def measure_time(time_unit: _unit = "ms", name: str | None = None) -> Callable:
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
                MEASUREMENTS[time_unit][name].append(parsed_duration)

            if logging.root.level == logging.INFO:
                logging.info(msg)

            elif logging.root.level == logging.DEBUG:
                msg = msg + f"\n{func_filename}"
                logging.debug(msg)

            return return_val

        return wrapper_function_measure_time

    return function_measure_time


def plot_time_measurements(
    names: list[str],
    time_unit: _unit = "ms",
    skip_first_n: int = 0,
    fill_outliers_with_nan: bool = True,
    filepath: str = "measurements.jpg",
):
    times = {
        k: v[skip_first_n:] for k, v in MEASUREMENTS[time_unit].items() if k in names
    }
    df = pd.DataFrame.from_dict(times)
    if fill_outliers_with_nan:
        for col in df.columns:
            q_low = df[col].quantile(0.01)
            q_hi = df[col].quantile(0.99)
            mask = ((df[col] < q_hi) & (df[col] > q_low)).values
            df[col][~mask] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    sns.histplot(
        df, element="step", kde=True, alpha=0.6, ax=axes[0], stat="density", bins=30
    )
    axes[0].set_xlabel(f"Latency [{time_unit}]", fontsize=14)
    axes[0].set_ylabel("Density", fontsize=14)

    for col in df.columns:
        sns.lineplot(x=df.index.values, y=df[col].values, label=col, ax=axes[1])
    axes[1].set_xlabel("Index", fontsize=14)
    axes[1].set_ylabel(f"Latency [{time_unit}]", fontsize=14)

    fig.savefig(filepath, bbox_inches="tight", dpi=300)
