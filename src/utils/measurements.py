from collections import defaultdict

from scipy.interpolate import interp1d

from src.monitoring.system import CKPT_MEMORY_MEASUREMENTS, MEMORY_MEASUREMENTS
from src.monitoring.time import TIME_MEASUREMENTS
from src.utils.utils import defaultdict2dict


def subtract_initial_memory_usage(
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


def interpolate_memory_to_time(
    time_measurements, memory_measurements, skip_first_n: int = 0
):
    _memory_measurements = defaultdict(lambda: defaultdict(list))
    for method_name, times in time_measurements.items():
        xnew = list(range(len(times)))
        for stat, values in memory_measurements[method_name].items():
            x = list(range(len(values)))
            y = values
            interp_fn = interp1d(x, y, fill_value="extrapolate")
            values_new = interp_fn(xnew)
            _memory_measurements[method_name][stat] = values_new[skip_first_n:].tolist()
    return _memory_measurements


def prepare_measurements(skip_first_n: int):
    time_measurements = TIME_MEASUREMENTS["ms"]
    memory_measurements = MEMORY_MEASUREMENTS
    ckpt_memory_measurements = CKPT_MEMORY_MEASUREMENTS
    memory_measurements = interpolate_memory_to_time(
        time_measurements, memory_measurements, skip_first_n
    )
    memory_measurements = subtract_initial_memory_usage(
        memory_measurements, ckpt_memory_measurements
    )
    time_measurements = {k: v[skip_first_n:] for k, v in time_measurements.items()}
    time_measurements = defaultdict2dict(time_measurements)
    memory_measurements = defaultdict2dict(memory_measurements)
    return time_measurements, memory_measurements
