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
    if t < 0.50:
        return 500*(np.pi/30)
    elif t < 1.0:
       return 1000.0*(np.pi/30)
    elif t < 1.50:
        return 2000.0*(np.pi/30)
    elif t < 2.0:
       return 3000.0*(np.pi/30)
    elif t < 2.5:
         return 4000.0*(np.pi/30)
    else:
        return 2000.0*(np.pi/30)
    
def generate_random_torque_profile(
    duration: float = 2.5,
    switch_interval: float = 0.1,
    min_torque: float = 0.0,
    max_torque: float = 0.05,
    seed: int = 42
) -> callable:
    """Generate a random step-changing torque load profile."""
    np.random.seed(seed)
    
    # Pre-generate torque values
    n_intervals = int(np.ceil(duration / switch_interval))
    torque_values = np.random.uniform(min_torque, max_torque, n_intervals)
    
    def load_func(t: float) -> float:
        if t < 0 or t >= duration:
            return 0.0
        interval_idx = min(int(t / switch_interval), n_intervals - 1)
        return float(torque_values[interval_idx])
    
    return load_func


def main():
    t_final = 3.0
    dt_sim=5e-5
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

    random_load = generate_random_torque_profile(
        duration=t_final,
        switch_interval=0.15,
        min_torque=0.0,
        max_torque=0.2,
        seed=123  # Change this for different random patterns
    )


    # Plant
    plant = PmsmPlant(
        Rs=Rs,
        Ld=Ld,
        Lq=Lq,
        psi_f=psi_f,
        p=p,
        J=J,
        B=B,
        #load_torque_func=load_torque_profile,
        load_torque_func=random_load,
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
