# pmsm_simulator.py
import numpy as np
import pandas as pd

from pmsm_plant import PmsmPlant
from foc_controller import FocCurrentController
from speed_controller import SpeedController
from typing import Optional, Callable
from sensor_noise import add_current_noise, speed_quantization
from ramp_gen import SimpleRamp

class PmsmFocSimulator:
    """
    PMSM + FOC simulator.

    Can run:
      - speed control (omega_ref_func + SpeedController)
    """

    def __init__(
        self,
        plant: PmsmPlant,
        current_controller: FocCurrentController,
        t_final: float = 0.5,
        # reference profiles:
        iq_ref_func: Optional[Callable[[float], float]] = None,     # used if no speed_ctrl
        id_ref_func: Optional[Callable[[float], float]] = None,
        omega_ref_func: Optional[Callable[[float], float]] = None,  # used if speed_ctrl
        speed_controller: Optional[SpeedController] = None,
        x0=None,
        dt_sim: float = 1e-5,
        dt_current: float = 5e-5,
        dt_speed: float = 5e-4,
    ):
        self.plant = plant
        self.current_controller = current_controller
        self.speed_controller = speed_controller
        self.dt_sim = dt_sim
        self.dt_current = dt_current
        self.dt_speed = dt_speed
        self.t_final = t_final

        self.iq_ref_func = iq_ref_func
        self.id_ref_func = id_ref_func or (lambda t: 0.0)
        self.omega_ref_func = omega_ref_func

        if x0 is None:
            self.x0 = np.array([0.0, 0.0, 0.0])
        else:
            self.x0 = np.array(x0, dtype=float)

        
            

    def run(self) -> pd.DataFrame:
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
        omega_ref = 0.0
        v_d, v_q = 0.0, 0.0

        # Track last update times
        last_current_update = -dt_current
        last_speed_update = -dt_speed

        for k in range(n_steps):
            t = k * dt_sim
            i_d, i_q, omega_m = x
            omega_e = self.plant.p * omega_m


            # Speed controller update
            if self.speed_controller is not None and (t - last_speed_update) >= dt_speed - 1e-12:
                omega_ref = self.omega_ref_func(t) if self.omega_ref_func is not None else 0.0
                iq_ref = self.speed_controller.step(omega_ref, omega_m)
                last_speed_update = t
            # Current controller update
            if (t - last_current_update) >= dt_current - 1e-12:
                if self.speed_controller is not None:
                    # Use iq_ref from speed controller
                    pass
                else:
                    iq_ref = self.iq_ref_func(t) if self.iq_ref_func is not None else 0.0
                id_ref = self.id_ref_func(t)
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

            # Add noisy measurements
            out["i_d_meas"] = add_current_noise(out["i_d"])
            out["i_q_meas"] = add_current_noise(out["i_q"])
            out["omega_m_meas"] = speed_quantization (out["omega_m"], 65536, dt_current)

            out.update({
                "t": t,
                "i_d_ref": self.id_ref_func(t),
                "i_q_ref": iq_ref,
                "v_d": v_d,
                "v_q": v_q,
                "omega_ref": omega_ref if self.speed_controller is not None else None,
            })
            logs.append(out)

        df = pd.DataFrame(logs)
        return df
