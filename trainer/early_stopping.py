from __future__ import annotations


class EarlyStopping:
    """
    Stop training when the monitored validation metric no longer improves.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'.")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_value: float | None = None
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "max":
            return value > self.best_value + self.min_delta

        return value < self.best_value - self.min_delta

    def step(self, value: float) -> bool:
        """
        Returns True when training should stop.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
