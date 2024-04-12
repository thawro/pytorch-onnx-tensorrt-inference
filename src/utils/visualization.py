from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from ..monitoring.system import CKPT_MEMORY_MEASUREMENTS, MEMORY_MEASUREMENTS
from ..monitoring.time import TIME_MEASUREMENTS, _time_unit

sns.set_style("whitegrid")


def plot_time_measurements(
    time_measurements,
    time_unit: _time_unit = "ms",
    fill_outliers_with_nan: bool = True,
    filepath: str = "time_measurements.jpg",
):
    df = pd.DataFrame.from_dict(time_measurements)
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


def interpolate_memory_to_time(time_unit: _time_unit = "ms", skip_first_n: int = 0):
    memory_measurements = defaultdict(lambda: defaultdict(list))
    for method_name, times in TIME_MEASUREMENTS[time_unit].items():
        xnew = list(range(len(times)))
        for stat, values in MEMORY_MEASUREMENTS[method_name].items():
            x = list(range(len(values)))
            y = values
            interp_fn = interp1d(x, y)
            values_new = interp_fn(xnew)
            memory_measurements[method_name][stat] = values_new[skip_first_n:]
    return memory_measurements


def plot_gpu_memory_measurements(
    names: list[str],
    device: str,
    time_unit: _time_unit = "ms",
    skip_first_n: int = 0,
    dirpath: str = "memory_measurements.jpg",
):
    memory_measurements = interpolate_memory_to_time(time_unit, skip_first_n)
    for engine_name, memory_ckpt_stats in CKPT_MEMORY_MEASUREMENTS.items():
        init_stats = memory_ckpt_stats["initialized"]
        # NOTE: remove init value of memory (mb) stats to know how much increased
        for stat_name, value in init_stats.items():
            if "mb" in stat_name:
                memory_measurements[engine_name][stat_name] -= value
    memory_measurements = {k: v for k, v in memory_measurements.items() if k in names}
    gpu_0_mb_measurements = {
        engine_name: memory_stats["gpu_0_mb"]
        for engine_name, memory_stats in memory_measurements.items()
    }
    gpu_0_util_measurements = {
        engine_name: memory_stats["gpu_0_pct_util"]
        for engine_name, memory_stats in memory_measurements.items()
    }
    gpu_0_mb_df = pd.DataFrame.from_dict(gpu_0_mb_measurements)
    gpu_0_util_df = pd.DataFrame.from_dict(gpu_0_util_measurements)

    stats = {"gpu_mb": gpu_0_mb_df, "gpu_util": gpu_0_util_df}
    for name, df in stats.items():
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        sns.histplot(
            df, element="step", kde=True, alpha=0.6, ax=axes[0], stat="density", bins=30
        )
        axes[0].set_xlabel(name, fontsize=14)
        axes[0].set_ylabel("Density", fontsize=14)

        for col in df.columns:
            sns.lineplot(x=df.index.values, y=df[col].values, label=col, ax=axes[1])
        axes[1].set_xlabel("Index", fontsize=14)
        axes[1].set_ylabel(name, fontsize=14)

        fig.savefig(
            f"{dirpath}/{device}_{name}_measurements.jpg", bbox_inches="tight", dpi=300
        )


def plot_measurements(dirpath: str, cuda_time_measurements, cpu_time_measurements):
    plot_time_measurements(
        cuda_time_measurements, filepath=f"{dirpath}/cuda_time_measurements.jpg"
    )
    plot_time_measurements(
        cpu_time_measurements, filepath=f"{dirpath}/cpu_time_measurements.jpg"
    )
    # plot_gpu_memory_measurements(names, device, time_unit, skip_first_n, dirpath)
