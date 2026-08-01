from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class AverageMeter:
    """Track running average values during training."""

    name: str
    value: float = 0.0
    average: float = 0.0
    total: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * n
        self.count += n
        self.average = self.total / max(self.count, 1)


class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self.start_time: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Call start() before elapsed().")
        return time.perf_counter() - self.start_time
