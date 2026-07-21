"""Run a workload under N concurrent workers and measure latency/throughput.

Used to simulate 2-3 concurrent connections against Whisper or Ollama and see
how per-request latency and throughput hold up under load.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class TaskOutcome:
    index: int
    latency_s: float
    ok: bool
    result: object = None
    error: str | None = None


@dataclass
class ConcurrencyResult:
    concurrency: int
    num_tasks: int
    wall_time_s: float
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def latencies(self) -> list[float]:
        return [o.latency_s for o in self.outcomes if o.ok]

    def _pct(self, p: float) -> float:
        vals = sorted(self.latencies)
        if not vals:
            return float("nan")
        k = min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))
        return vals[k]

    @property
    def mean_latency_s(self) -> float:
        v = self.latencies
        return sum(v) / len(v) if v else float("nan")

    @property
    def p50_latency_s(self) -> float:
        return self._pct(50)

    @property
    def p95_latency_s(self) -> float:
        return self._pct(95)

    @property
    def throughput_per_s(self) -> float:
        return len(self.latencies) / self.wall_time_s if self.wall_time_s else float("nan")

    @property
    def num_errors(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)

    def as_dict(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "num_tasks": self.num_tasks,
            "wall_time_s": round(self.wall_time_s, 2),
            "mean_latency_s": round(self.mean_latency_s, 3),
            "p50_latency_s": round(self.p50_latency_s, 3),
            "p95_latency_s": round(self.p95_latency_s, 3),
            "throughput_per_s": round(self.throughput_per_s, 3),
            "num_errors": self.num_errors,
        }


def run_concurrent(fn: Callable[[T], object], items: list[T], concurrency: int) -> ConcurrencyResult:
    """Apply `fn` to each item across `concurrency` worker threads.

    Records per-item latency and overall wall time. Exceptions are captured per
    item so one failure doesn't abort the run.
    """
    outcomes: list[TaskOutcome] = [None] * len(items)  # type: ignore

    def _wrapped(idx_item):
        idx, item = idx_item
        t0 = time.monotonic()
        try:
            res = fn(item)
            return TaskOutcome(idx, time.monotonic() - t0, True, res)
        except Exception as exc:
            return TaskOutcome(idx, time.monotonic() - t0, False, None, f"{type(exc).__name__}: {exc}")

    wall0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for outcome in pool.map(_wrapped, enumerate(items)):
            outcomes[outcome.index] = outcome
    wall = time.monotonic() - wall0

    return ConcurrencyResult(
        concurrency=concurrency,
        num_tasks=len(items),
        wall_time_s=wall,
        outcomes=outcomes,
    )
