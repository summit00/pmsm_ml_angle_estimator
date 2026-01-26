""""Ramp generator module."""
import numpy as np


class SimpleRamp:
    """Minimal ramp generator for smooth transitions."""

    def __init__(self, ramp_rate: float = 10.0, dt: float = 0.001, start: float = 0.0) -> None:
        """Initialize ramp generator."""
        self.ramp_rate = ramp_rate  # max change per second
        self.dt = dt
        self.current = start
        self.target = start

    def set_target(self, target: float) -> None:
        """Set new target value."""
        self.target = target

    def update(self) -> float:
        """Update and return current ramped value."""
        if self.current == self.target:
            return self.current

        step = self.ramp_rate * self.dt
        error = self.target - self.current

        if abs(error) <= step:
            self.current = self.target  # reached target
        else:
            self.current += np.sign(error) * step

        return self.current
