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

def speed_quantization(speed_rad_s, increments, sample_time=0.001):
    """
    Speed quantization.
    """
    # Basic quantization
    rad_per_pulse = 2 * np.pi / increments
    pulses_per_second = speed_rad_s / rad_per_pulse
    
    # Quantize pulse count (simulate integer counting)
    quantized_pulses = np.round(pulses_per_second * sample_time) / sample_time
    
    # Convert back to rad/s
    measured_speed = quantized_pulses * rad_per_pulse
    
    # Add small noise (optional)
    # measured_speed += np.random.normal(0, 0.01) * rad_per_pulse
    
    return measured_speed