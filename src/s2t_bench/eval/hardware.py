"""CPU + RAM monitoring during evaluation (no GPU).

Samples system CPU%, system RAM, this process's RSS, and optionally a named
external process (e.g. the `ollama` server, which is where Gemma inference runs)
on a background thread. Use as a context manager around the work you're timing.

    with HardwareMonitor(track_process_names=["ollama"]) as hw:
        ... run the workload ...
    print(hw.summary())
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class HardwareSummary:
    duration_s: float
    cpu_percent_mean: float
    cpu_percent_peak: float
    system_ram_used_peak_mb: float
    process_rss_peak_mb: float
    tracked_rss_peak_mb: dict  # name -> peak RSS MB
    num_cpus: int
    total_ram_mb: float
    num_samples: int

    def as_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "cpu_percent_mean": round(self.cpu_percent_mean, 1),
            "cpu_percent_peak": round(self.cpu_percent_peak, 1),
            "system_ram_used_peak_mb": round(self.system_ram_used_peak_mb, 1),
            "process_rss_peak_mb": round(self.process_rss_peak_mb, 1),
            "tracked_rss_peak_mb": {k: round(v, 1) for k, v in self.tracked_rss_peak_mb.items()},
            "num_cpus": self.num_cpus,
            "total_ram_mb": round(self.total_ram_mb, 1),
            "num_samples": self.num_samples,
        }


class HardwareMonitor:
    def __init__(self, interval_s: float = 0.25, track_process_names: list[str] | None = None) -> None:
        self.interval_s = interval_s
        self.track_process_names = [n.lower() for n in (track_process_names or [])]
        self._cpu: list[float] = []
        self._sys_ram: list[float] = []
        self._proc_rss: list[float] = []
        self._tracked: dict[str, list[float]] = {n: [] for n in self.track_process_names}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0 = 0.0
        self._t1 = 0.0

    def _find_tracked_rss(self, psutil) -> dict[str, float]:
        out = {n: 0.0 for n in self.track_process_names}
        if not self.track_process_names:
            return out
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info["name"] or "").lower()
                for target in self.track_process_names:
                    if target in name:
                        out[target] += proc.memory_info().rss / 1e6
            except Exception:
                continue
        return out

    def _sample_once(self, psutil, proc) -> None:
        self._cpu.append(psutil.cpu_percent(interval=None))
        self._sys_ram.append(psutil.virtual_memory().used / 1e6)
        self._proc_rss.append(proc.memory_info().rss / 1e6)
        for name, rss in self._find_tracked_rss(psutil).items():
            self._tracked[name].append(rss)

    def _run(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            self._sample_once(psutil, proc)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "HardwareMonitor":
        import psutil

        self._t0 = time.monotonic()
        psutil.cpu_percent(interval=None)  # prime the counter
        # One synchronous sample so even a very fast workload has data.
        self._sample_once(psutil, psutil.Process())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._t1 = time.monotonic()

    def summary(self) -> HardwareSummary:
        import psutil

        def _mean(v):
            return sum(v) / len(v) if v else 0.0

        def _peak(v):
            return max(v) if v else 0.0

        return HardwareSummary(
            duration_s=self._t1 - self._t0,
            cpu_percent_mean=_mean(self._cpu),
            cpu_percent_peak=_peak(self._cpu),
            system_ram_used_peak_mb=_peak(self._sys_ram),
            process_rss_peak_mb=_peak(self._proc_rss),
            tracked_rss_peak_mb={k: _peak(v) for k, v in self._tracked.items()},
            num_cpus=psutil.cpu_count(logical=True) or 0,
            total_ram_mb=psutil.virtual_memory().total / 1e6,
            num_samples=len(self._cpu),
        )
