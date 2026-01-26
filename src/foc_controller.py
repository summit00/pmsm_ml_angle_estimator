"""FOC current controller in dq frame with decoupling."""

import numpy as np
from numpy.typing import NDArray
from pid_controller import PIDController


class FocCurrentController:
    """FOC current controller in dq frame with decoupling."""

    def __init__(
        self,
        dt: float,
        # PI gains:
        kp_d: float = 5.0,
        ki_d: float = 1000.0,
        kp_q: float = 5.0,
        ki_q: float = 1000.0,
        # machine params for decoupling:
        l_d: float = 1e-3,
        l_q: float = 1.2e-3,
        psi_f: float = 0.05,
        p: int = 3,
        # voltage limit:
        v_limit: float = 50.0,
    ):
        """Initialize FOC current controller."""
        self.dt = dt
        self.v_limit = v_limit

        # motor params
        self.l_d = l_d
        self.l_q = l_q
        self.psi_f = psi_f
        self.p = p  # not strictly needed here, but handy if extended later

        # inner PI regulators
        self.pid_d = PIDController(kp=kp_d, ki=ki_d, kd=0.0, dt=dt)
        self.pid_q = PIDController(kp=kp_q, ki=ki_q, kd=0.0, dt=dt)

        self.pid_d.set_output_limits(-v_limit, v_limit)
        self.pid_q.set_output_limits(-v_limit, v_limit)

    def get_params(self) -> NDArray[np.float64]:
        """Return gains as a flat numpy array [kp_d, ki_d, kp_q, ki_q]."""
        return np.array(
            [
                self.pid_d.kp,
                self.pid_d.ki,
                self.pid_q.kp,
                self.pid_q.ki,
            ],
            dtype=float,
        )

    def set_params(self, gains: NDArray[np.float64]) -> None:
        """Set gains from a flat array [kp_d, ki_d, kp_q, ki_q]."""
        gains = np.asarray(gains, dtype=float)
        self.pid_d.kp = gains[0]
        self.pid_d.ki = gains[1]
        self.pid_q.kp = gains[2]
        self.pid_q.ki = gains[3]

    def reset(self) -> None:
        """Reset internal states of the controller."""
        self.pid_d.reset()
        self.pid_q.reset()

    def step(
        self,
        i_d_ref: float,
        i_q_ref: float,
        i_d_meas: float,
        i_q_meas: float,
        omega_e: float,
    ) -> tuple[float, float]:
        """Compute v_d, v_q from current references and measurements, with decoupling.

        Args:
            i_d_ref: current reference for d-axis
            i_q_ref: current reference for q-axis
            i_d_meas: measured current in d-axis
            i_q_meas: measured current in q-axis
            omega_e: electrical angular speed [rad/s]

        Returns:
            (v_d, v_q)
        """
        # PI outputs (base voltages, no decoupling yet)
        v_d_pi = self.pid_d.update(i_d_ref, i_d_meas)
        v_q_pi = self.pid_q.update(i_q_ref, i_q_meas)

        # decoupling feedforward terms
        v_d_dec = -omega_e * self.l_q * i_q_meas
        v_q_dec = +omega_e * self.l_d * i_d_meas + omega_e * self.psi_f

        # sum PI + decoupling
        v_d = v_d_pi + v_d_dec
        v_q = v_q_pi + v_q_dec

        # voltage magnitude limiting (circle limitation)
        v_mag = np.sqrt(v_d * v_d + v_q * v_q)
        if v_mag > self.v_limit and v_mag > 0.0:
            scale = self.v_limit / v_mag
            v_d *= scale
            v_q *= scale

        return v_d, v_q
