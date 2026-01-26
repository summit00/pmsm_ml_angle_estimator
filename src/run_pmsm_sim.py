"""Run PMSM FOC simulation with random torque and speed profiles."""
import time

import numpy as np
import pandas as pd

from foc_controller import FocCurrentController
from pmsm_plant import PmsmPlant
from pmsm_plotter import plot_pmsm_results
from pmsm_simulator import PmsmFocSimulator
from speed_controller import SpeedController


def generate_random_torque_profile(
    duration: float = 2.5,
    switch_interval: float = 0.1,
    min_torque: float = 0.0,
    max_torque: float = 0.05,
    seed: int = 42,
) -> callable:
    """Generate a random step-changing torque load profile."""
    np.random.seed(seed)

    # Pre-generate torque values
    n_intervals = int(np.ceil(duration / switch_interval))
    torque_values = np.random.uniform(min_torque, max_torque, n_intervals)

    def load_func(t: float) -> float:
        if t < 0.1 or t >= duration:
            return 0.0
        interval_idx = min(int(t / switch_interval), n_intervals - 1)
        return float(torque_values[interval_idx])

    return load_func


def generate_random_speed_profile(
    duration: float = 2.5,
    switch_interval: float = 0.4,
    min_rpm: float = 500.0,
    max_rpm: float = 4000.0,
    seed: int = 42,
) -> callable:
    """Generate a random step-changing speed reference profile in rad/s."""
    np.random.seed(seed)

    # Pre-generate RPM values
    n_intervals = int(np.ceil(duration / switch_interval))
    rpm_values = np.random.uniform(min_rpm, max_rpm, n_intervals)

    # Generate final RPM value (for t >= duration)
    final_rpm = np.random.uniform(min_rpm, max_rpm)

    def omega_ref_profile(t: float) -> float:
        if t < 0.0:
            return float(rpm_values[0]) * (np.pi / 30)
        if t >= duration:
            return final_rpm * (np.pi / 30)

        interval_idx = min(int(t / switch_interval), n_intervals - 1)
        return float(rpm_values[interval_idx]) * (np.pi / 30)

    return omega_ref_profile


def main() -> None:
    """Main sim function."""
    t_final = 1.0
    dt_sim = 5e-5
    dt_current = 5e-5
    dt_speed = 2.5e-4

    # Motor & FOC parameters (shared for plant + controller)
    r_s = 0.315
    l_d = 3e-4
    l_q = 2.8e-4
    psi_f = 0.0107
    p = 3
    j = 0.0000075
    b = 0

    random_load = generate_random_torque_profile(
        duration=t_final,
        switch_interval=0.15,
        min_torque=0.0,
        max_torque=0.2,
        seed=123,  # Change this for different random patterns
    )

    random_speed = generate_random_speed_profile(
        duration=t_final,
        switch_interval=0.5,
        min_rpm=150.0,
        max_rpm=12000.0,
    )

    # Plant - note: initial state now includes angle [i_d, i_q, theta_m, omega_m]
    plant = PmsmPlant(
        r_s=r_s,
        l_d=l_d,
        l_q=l_q,
        psi_f=psi_f,
        p=p,
        j=j,
        b=b,
        load_torque_func=random_load,
    )

    # Speed controller (PI)
    speed_ctrl = SpeedController(
        dt=dt_speed,
        kp_w=0.15,
        ki_w=0.3,
        iq_limit=5.0,
    )

    # FOC current controller
    foc = FocCurrentController(
        dt=dt_current,
        kp_d=0.4,
        ki_d=300,
        kp_q=0.4,
        ki_q=300,
        l_d=l_d,
        l_q=l_q,
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
        # omega_ref_func=omega_ref_profile,
        omega_ref_func=random_speed,
        dt_sim=dt_sim,
        dt_current=dt_current,
        dt_speed=dt_speed,
        ramp_rate=100000.0 * (np.pi / 30),
    )
    start = time.perf_counter()
    df = sim.run()
    end = time.perf_counter()
    print(f"Simulation time: {end - start:.3f} s")
    print("Data Collected: ", len(df), "rows")

    # Create ML-ready dataset for angle estimation
    # Features: everything we can measure (noisy/quantized)
    # Target: true rotor angle
    ml_df = pd.DataFrame(
        {
            "time": df["t"],
            "i_d": df["i_d"],
            "i_q": df["i_q"],
            "i_alpha": df["i_alpha"],
            "i_beta": df["i_beta"],
            "v_d": df["v_d"],
            "v_q": df["v_q"],
            "v_alpha": df["v_alpha"],
            "v_beta": df["v_beta"],
            "sin_theta_e": df["sin_theta_e"],
            "cos_theta_e": df["cos_theta_e"],
            "omega_e": df["omega_e"],  # True mechanical speed
            "theta_e": df["theta_e"],  # True electrical angle [wrapped]
            "i_q_ref": df["i_q_ref"],
            "i_d_ref": df["i_d_ref"],
            "omega_ref": df["omega_ref"],
            "torque_e": df["torque_e"],
            "torque_load": df["torque_load"],
            "i_alpha_meas_12": df["i_alpha_meas_12"],
            "i_beta_meas_12": df["i_beta_meas_12"],
            "i_alpha_meas_8": df["i_alpha_meas_8"],
            "i_beta_meas_8": df["i_beta_meas_8"],
            "sin_theta_e_12bit": df["sin_theta_e_12bit"],
            "cos_theta_e_12bit": df["cos_theta_e_12bit"],
            "sin_theta_e_8bit": df["sin_theta_e_8bit"],
            "cos_theta_e_8bit": df["cos_theta_e_8bit"],
        }
    )

    # Save ML dataset
    ml_df.to_csv("data/raw/angle_estimation_dataset.csv", index=False)

    # Plot results
    plot_pmsm_results(df.iloc[::5])


if __name__ == "__main__":
    main()
