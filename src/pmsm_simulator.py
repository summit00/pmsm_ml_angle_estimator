"""PMSM FOC simulator module."""
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from foc_controller import FocCurrentController
from pmsm_plant import PmsmPlant
from ramp_gen import SimpleRamp
from sensors import add_current_noise, quantize_angle
from speed_controller import SpeedController
from transform import ClarkeParkTransform


class PmsmFocSimulator:
    """PMSM + FOC simulator."""

    def __init__(
        self,
        plant: PmsmPlant,
        current_controller: FocCurrentController,
        speed_controller: SpeedController,
        t_final: float = 0.5,
        omega_ref_func: Callable[[float], float] | None = None,
        x0: Sequence[float] | None = None,
        dt_sim: float = 1e-5,
        dt_current: float = 5e-5,
        dt_speed: float = 5e-4,
        ramp_rate: float = 1000.0,
    ) -> None:
        """Initialize PMSM FOC simulator."""
        self.plant = plant
        self.current_controller = current_controller
        self.speed_controller = speed_controller
        self.dt_sim = dt_sim
        self.dt_current = dt_current
        self.dt_speed = dt_speed
        self.t_final = t_final
        self.omega_ref_func = omega_ref_func
        self.ramp_rate = ramp_rate

        if x0 is None:
            self.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.x0 = np.array(x0, dtype=float)

    def run(self) -> pd.DataFrame:
        """Run the PMSM FOC simulation."""
        dt_sim = self.dt_sim
        dt_current = self.dt_current
        dt_speed = self.dt_speed
        n_steps = int(self.t_final / dt_sim)
        x = self.x0.copy()

        self.current_controller.reset()
        if self.speed_controller is not None:
            self.speed_controller.reset()

        logs = []

        # Initialize controller states
        iq_ref = 0.0
        v_d, v_q = 0.0, 0.0
        v_alpha, v_beta = 0.0, 0.0
        i_alpha, i_beta = 0.0, 0.0

        omega_ref_ramped = 0.0

        # Track last update times
        last_current_update = -dt_current
        last_speed_update = -dt_speed

        ramp = SimpleRamp(ramp_rate=self.ramp_rate, dt=dt_speed, start=0.0)

        transform = ClarkeParkTransform()

        for k in range(n_steps):
            t = k * dt_sim
            i_d, i_q, theta_m, omega_m = x
            omega_e = self.plant.p * omega_m

            # Speed controller update
            if (
                self.speed_controller is not None
                and (t - last_speed_update) >= dt_speed - 1e-12
            ):
                raw_omega_ref = self.omega_ref_func(t)
                ramp.set_target(raw_omega_ref)
                omega_ref_ramped = ramp.update()
                iq_ref = self.speed_controller.step(omega_ref_ramped, omega_e)
                # omega_ref = self.omega_ref_func(t)
                # iq_ref = self.speed_controller.step(omega_ref, omega_m)
                last_speed_update = t
            # Current controller update
            if (t - last_current_update) >= dt_current - 1e-12:
                id_ref = 0.0  # always zero d-axis current reference.
                # transform to alpha-beta frame for logging
                v_alpha, v_beta = transform.inverse_park_transform(
                    v_d, v_q, theta_m * self.plant.p
                )
                i_alpha, i_beta = transform.inverse_park_transform(
                    i_d, i_q, theta_m * self.plant.p
                )
                v_d, v_q = self.current_controller.step(
                    i_d_ref=id_ref,
                    i_q_ref=iq_ref,
                    i_d_meas=i_d,
                    i_q_meas=i_q,
                    omega_e=omega_e,
                )
                last_current_update = t
            # Plant integration (Euler)
            dxdt = self.plant.ode(t, x, np.array([v_d, v_q]))
            x = x + dxdt * dt_sim

            out = self.plant.output(t, x)

            out["i_alpha_meas_12"] = add_current_noise(i_alpha, adc_bits=12)
            out["i_beta_meas_12"] = add_current_noise(i_beta, adc_bits=12)
            out["i_alpha_meas_8"] = add_current_noise(i_alpha, adc_bits=8)
            out["i_beta_meas_8"] = add_current_noise(i_beta, adc_bits=8)

            out.update(
                {
                    "t": t,
                    "i_d_ref": 0.0,
                    "i_q_ref": iq_ref,
                    "v_d": v_d,
                    "v_q": v_q,
                    "v_alpha": v_alpha,
                    "v_beta": v_beta,
                    "i_alpha": i_alpha,
                    "i_beta": i_beta,
                    "sin_theta_e": np.sin(theta_m * self.plant.p),
                    "cos_theta_e": np.cos(theta_m * self.plant.p),
                    "sin_theta_e_12bit": quantize_angle(
                        np.sin(theta_m * self.plant.p), encoder_bits=12
                    ),
                    "cos_theta_e_12bit": quantize_angle(
                        np.cos(theta_m * self.plant.p), encoder_bits=12
                    ),
                    "sin_theta_e_8bit": quantize_angle(
                        np.sin(theta_m * self.plant.p), encoder_bits=8
                    ),
                    "cos_theta_e_8bit": quantize_angle(
                        np.cos(theta_m * self.plant.p), encoder_bits=8
                    ),
                    "omega_ref": omega_ref_ramped,
                }
            )
            logs.append(out)

        df = pd.DataFrame(logs)
        return df
