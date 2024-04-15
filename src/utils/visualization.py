from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.monitoring.time import _time_unit

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


def plot_memory_measurements(
    memory_measurements, stats_names: list[str], dirpath: str = "."
):
    stats_dfs = {}
    for stat_name in stats_names:
        stat_measurements = {
            engine_name: memory_stats[stat_name]
            for engine_name, memory_stats in memory_measurements.items()
        }
        stats_dfs[stat_name] = pd.DataFrame.from_dict(stat_measurements)

    for name, df in stats_dfs.items():
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

        fig.savefig(f"{dirpath}/{name}_measurements.jpg", bbox_inches="tight", dpi=300)


def plot_measurements(
    dirpath: str,
    cpu_time_measurements,
    cpu_memory_measurements,
    cuda_time_measurements,
    cuda_memory_measurements,
):
    Path(dirpath).mkdir(exist_ok=True, parents=True)
    gpu_stats_names = ["gpu_0_mb", "gpu_0_pct_util"]
    cpu_stats_names = ["cpu_mb", "cpu_pct_util"]
    plot_memory_measurements(cpu_memory_measurements, cpu_stats_names, dirpath)
    plot_memory_measurements(cuda_memory_measurements, gpu_stats_names, dirpath)
    plot_time_measurements(
        cuda_time_measurements, filepath=f"{dirpath}/cuda_time_measurements.jpg"
    )
    plot_time_measurements(
        cpu_time_measurements, filepath=f"{dirpath}/cpu_time_measurements.jpg"
    )
