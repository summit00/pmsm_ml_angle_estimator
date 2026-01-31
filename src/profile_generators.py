import numpy as np
from collections.abc import Callable

def generate_random_torque_profile(
    duration: float = 2.5,
    switch_interval: float = 0.1,
    min_torque: float = 0.0,
    max_torque: float = 0.05,
    seed: int = 42,
) -> Callable[[float], float]:
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
) -> Callable[[float], float]:
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