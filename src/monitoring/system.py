"""Class for monitoring system stats."""

import logging
import threading
from collections import defaultdict

import psutil
from typing import List, Optional, Union, Dict
from jtop import jtop



MEMORY_MEASUREMENTS = defaultdict(lambda: defaultdict(list))
CKPT_MEMORY_MEASUREMENTS = defaultdict(dict)


def gather_system_metrics() -> Dict[str, float]:
	metrics = {}
	cpu_percent = psutil.cpu_percent()
	cpu_memory = psutil.virtual_memory()
	metrics["cpu_pct_util"] = cpu_percent
	metrics["cpu_mb"] = cpu_memory.used / 1e6
	ram_used = -1
	gpu_util = -1
	with jtop() as jetson:
		if jetson.ok():
			stats = jetson.stats
			gpu_memory_kb = stats['RAM']
			temp_CPU = stats['Temp CPU']
			temp_GPU = stats['Temp GPU']
			power_cur = stats['power cur']
			gpu_util = jetson.gpu['val']
	metrics[f"gpu_mb"] = float(gpu_memory_kb) / 1e3
	metrics[f"gpu_pct_util"] = gpu_util        
	return metrics


class SystemMetricsCollector:
    """Class for monitoring GPU stats."""

    def __init__(self):
        self._metrics = defaultdict(list)

    @property
    def current_metrics(self) -> Dict[str, float]:
        return gather_system_metrics()

    def collect_metrics(self):
        for name, value in self.current_metrics.items():
            self._metrics[name].append(value)

    @property
    def metrics(self):
        return self._metrics

    def clear_metrics(self):
        self._metrics.clear()

    def aggregate_metrics(self) -> Dict[str, float]:
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}


class SystemMetricsMonitor:
    """Class for monitoring system stats."""

    def __init__(
        self,
        sampling_interval: float = 0.001,
        samples_before_logging: int = 1,
        name: Optional[str] = None,
    ):
        self.sampling_interval = sampling_interval
        self._shutdown_event = threading.Event()
        self._process = None
        self.metrics_collector = SystemMetricsCollector()
        self.samples_before_logging = samples_before_logging
        self.ckpt_metrics = {}
        self.name = name
        self.checkpoint_metrics("initialized")

    def reinitialize(self, name: Optional[str] = None):
        return SystemMetricsMonitor(
            sampling_interval=self.sampling_interval,
            samples_before_logging=self.samples_before_logging,
            name=name,
        )

    @property
    def name_info(self) -> str:
        return f"[{self.name}] "

    def start(self):
        """Start monitoring system metrics."""
        try:
            self._process = threading.Thread(
                target=self._monitor,
                daemon=True,
                name=self.name,
            )
            self._process.start()
            logging.info(f"{self.name_info}Started.")
        except Exception as e:
            logging.warning(f"{self.name_info}Failed to start: {e}")
            self._process = None

    def checkpoint_metrics(self, name: str):
        current_metrics = self.metrics_collector.current_metrics
        self.ckpt_metrics[name] = current_metrics
        CKPT_MEMORY_MEASUREMENTS[self.name][name] = current_metrics

    def _monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        while not self._shutdown_event.is_set():
            self.monitor()

    def monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        for _ in range(self.samples_before_logging):
            self.collect_metrics()
            self._shutdown_event.wait(self.sampling_interval)
        metrics = self.aggregate_metrics()
        try:
            self.publish_metrics(metrics)
        except Exception as e:
            logging.exception(f"{self.name_info}Failed to log system metrics: {e}")
            return

    def collect_metrics(self):
        """Collect system metrics."""
        self.metrics_collector.collect_metrics()
        metrics = self.metrics_collector._metrics
        return metrics

    def aggregate_metrics(self):
        """Aggregate collected metrics."""
        metrics = self.metrics_collector.aggregate_metrics()
        return metrics

    def publish_metrics(self, metrics: Dict[str, float]):
        """Do something with collected metrics and clear them."""
        for name, value in metrics.items():
            MEMORY_MEASUREMENTS[self.name][name].append(value)
        self.metrics_collector.clear_metrics()

    def finish(self):
        """Stop monitoring system metrics."""
        if self._process is None:
            return
        logging.info(f"{self.name_info} Stopping System Monitor")
        self._shutdown_event.set()
        self._process.join()
        self._process = None
