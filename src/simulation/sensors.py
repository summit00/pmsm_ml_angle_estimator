import numpy as np


def add_current_noise(i_true: float, 
                     adc_bits: int = 12, 
                     range_min: float = -10.0, 
                     range_max: float = 10.0,
                     noise_std: float = 0.02) -> float:
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