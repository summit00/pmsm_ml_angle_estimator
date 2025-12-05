# run_pmsm_sim.py
from pmsm_plant import PmsmPlant
from foc_controller import FocCurrentController
from speed_controller import SpeedController
from pmsm_simulator import PmsmFocSimulator
from pmsm_plotter import plot_pmsm_results
import numpy as np
import time
import pandas as pd


def omega_ref_profile(t: float) -> float:
    if t < 1.50:
        return 500*(np.pi/30)
    # elif t < 0.80:
    #     return 1000.0*(np.pi/30)
    # elif t < 1.20:
    #     return 1000.0*(np.pi/30)
    # elif t < 1.60:
    #     return 2000.0*(np.pi/30)
    # elif t < 2.0:
    #     return 4000.0*(np.pi/30)
    else:
        return 0.0

def load_torque_profile(t: float) -> float:
    if t < 0.10:
        return 0.001
    elif t < 0.20:
        return 0.005
    elif t < 0.30:
        return 0.008
    elif t < 0.40:
        return 0.01
    elif t < 0.50:
        return 0.03
    else:
        return 0.05


def main():
    t_final = 0.5
    dt_sim=1e-6
    dt_current=5e-5
    dt_speed=2.5e-4

    # Motor & FOC parameters (shared for plant + controller)
    Rs = 0.315
    Ld = 3e-4
    Lq = 2.8e-4
    psi_f = 0.0107
    p = 3
    J = 0.0000075
    B = 0

    # Plant
    plant = PmsmPlant(
        Rs=Rs,
        Ld=Ld,
        Lq=Lq,
        psi_f=psi_f,
        p=p,
        J=J,
        B=B,
        load_torque_func=load_torque_profile,
    )

    # Speed controller (PI)
    speed_ctrl = SpeedController(
        dt=dt_speed,
        kp_w=0.4,
        ki_w=1,
        iq_limit=5.0,
    )

    # FOC current controller
    foc = FocCurrentController(
        dt=dt_current,
        kp_d=0.4,
        ki_d=300,
        kp_q=0.4,
        ki_q=300,
        Ld=Ld,
        Lq=Lq,
        psi_f=psi_f,
        p=p,
        v_limit=24.0,
    )

    # Simulator
    sim = PmsmFocSimulator(
            plant=plant,
            current_controller=foc,
            speed_controller=speed_ctrl,
            t_final=t_final,
            omega_ref_func=omega_ref_profile,
            dt_sim=dt_sim,
            dt_current=dt_current,
            dt_speed=dt_speed,
            ramp_rate=50000.0*(np.pi/30),
    )
    start = time.perf_counter()
    df = sim.run()
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.3f} s")
    print("Data Collected: ", len(df), "rows")
    #df.to_csv("data/raw/pmsm_foc_simulation.csv")
    print(df.head())
    plot_pmsm_results(df.iloc[::2])  


if __name__ == "__main__":
    main()
