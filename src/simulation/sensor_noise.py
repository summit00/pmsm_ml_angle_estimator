# sensor_noise.py
"""
Minimal sensor noise functions for PMSM simulation.
"""

import numpy as np


def add_current_noise(i_true: float, 
                     adc_bits: int = 12, 
                     range_min: float = -10.0, 
                     range_max: float = 10.0,
                     noise_std: float = 0.01) -> float:
    """
    Add ADC quantization and noise to current measurement.
    """
    # Add noise
    i_noisy = i_true + np.random.normal(0, noise_std)
    
    # Quantize
    lsb = (range_max - range_min) / (2**adc_bits - 1)
    i_clipped = np.clip(i_noisy, range_min, range_max)
    i_quantized = np.round((i_clipped - range_min) / lsb) * lsb + range_min
    
    return float(i_quantized)

def speed_quantization(ideal_speed_rads, encoder_resolution=4096, update_rate_hz=10000):
    """
    Speed quantization.
    """
    # Convert resolution to pulses per radian
    pulses_per_radian = encoder_resolution / (2 * np.pi)
    
    # Time step between encoder readings
    dt = 1.0 / update_rate_hz
    
    # Ideal position change in this time step
    ideal_position_change = ideal_speed_rads * dt
    
    # Quantize the position change to nearest encoder pulse
    ideal_pulses = ideal_position_change * pulses_per_radian
    quantized_pulses = np.round(ideal_pulses)
    
    # Add small random noise (±1 count jitter)
    noisy_pulses = quantized_pulses + np.random.randint(-1, 2)
    
    # Convert back to speed measurement
    measured_position_change = noisy_pulses / pulses_per_radian
    measured_speed = measured_position_change / dt
    
    return measured_speed

def quantize_speed_with_encoder(speed_rad_s, encoder_increments, timestep):
    """
    Quantizes a continuous speed signal (rad/s) using an ABI encoder model.
    
    Parameters
    ----------
    speed_rad_s : float
        True continuous mechanical speed in rad/s.
    encoder_increments : int
        Number of electrical or mechanical counts per revolution (PPR * 4 typically for ABI).
    timestep : float
        Simulation timestep in seconds.
    
    Returns
    -------
    float
        Quantized speed in rad/s.
    """
    
    # True angle change during timestep
    delta_theta = speed_rad_s * timestep

    # Convert to encoder counts (continuous)
    counts_per_rev = encoder_increments
    counts_f = delta_theta / (2 * np.pi) * counts_per_rev

    # Quantize counts to nearest integer
    counts_q = np.round(counts_f)

    # Convert back to a quantized angle change
    delta_theta_q = counts_q * (2 * np.pi / counts_per_rev)

    # Quantized speed estimate
    speed_q = delta_theta_q / timestep

    return speed_q