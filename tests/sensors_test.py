import pytest
from sensors import add_current_noise, quantize_angle
import numpy as np

def test_add_current_noise_quantization() -> None:
    """Test that quantization maps to the correct LSB steps without noise."""
    val = add_current_noise(5.0, adc_bits=12, noise_std=0)
    lsb = 20 / 4095
    expected = np.round((5.0 - (-10.0)) / lsb) * lsb + (-10.0)
    assert val == pytest.approx(expected, abs=1e-6)

def test_add_current_noise_clipping() -> None:
    """Test that values outside the range are clipped."""
    val_high = add_current_noise(100.0, range_max=10.0, noise_std=0)
    val_low = add_current_noise(-100.0, range_min=-10.0, noise_std=0)
    assert val_high == 10.0
    assert val_low == -10.0

def test_quantisize_angle() -> None:
    """Test that angle quantization works correctly."""
    angle = np.pi / 4  # 45 degrees
    quantized = quantize_angle(angle, encoder_bits=12)
    encoder_counts = 2**12
    expected = np.round(angle * encoder_counts / (2 * np.pi)) * (2 * np.pi / encoder_counts)
    assert quantized == pytest.approx(expected, abs=1e-6)
