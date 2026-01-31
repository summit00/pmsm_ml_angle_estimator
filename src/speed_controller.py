"""Speed controller module."""

import numpy as np
from numpy.typing import NDArray

from pid_controller import PIDController


class SpeedController:
    """Simple speed controller using a PI controller."""

    def __init__(
        self,
        dt: float,
        kp_w: float = 1.0,
        ki_w: float = 100.0,
        iq_limit: float = 50.0,
    ) -> None:
        """Initialize speed controller."""
        self.dt = dt
        self.iq_limit = iq_limit

        self.pid_w = PIDController(kp=kp_w, ki=ki_w, kd=0.0, dt=dt)
        self.pid_w.set_output_limits(-iq_limit, iq_limit)

    def reset(self) -> None:
        """Reset Pid State."""
        self.pid_w.reset()

    def get_params(self) -> NDArray[np.float64]:
        """Return speed controller gains [kp_w, ki_w]."""
        return np.array([self.pid_w.kp, self.pid_w.ki], dtype=float)

    def set_params(self, theta: NDArray[np.float64] | float) -> None:
        """Set speed gains from [kp_w, ki_w]."""
        theta = np.asarray(theta, dtype=float)
        self.pid_w.kp = theta[0]
        self.pid_w.ki = theta[1]

    def step(self, omega_ref: float, omega_meas: float) -> float:
        """Returns i_q_ref to be tracked by the inner current loop."""
        iq_ref = self.pid_w.update(omega_ref, omega_meas)
        return float(iq_ref)
