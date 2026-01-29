"""Unit tests for the transform class using pytest framework."""

import pytest
from speed_controller import SpeedController
from pid_controller import PIDController


@pytest.fixture
def speed_controller() -> SpeedController:
    """Fixture that provides a default PID controller instance."""
    return SpeedController(dt= 1e-4, kp_w=0.5, ki_w=0.1, iq_limit=10.0)

def test_initialization(speed_controller: SpeedController) -> None:
    """Test that the controller initializes with the correct parameters."""
    assert speed_controller.dt == 1e-4
    assert speed_controller.iq_limit == 10.0
    assert speed_controller.pid_w.kp == 0.5
    assert speed_controller.pid_w.ki == 0.1
    assert speed_controller.pid_w.dt == 1e-4
    assert speed_controller.pid_w.min_output == -10.0
    assert speed_controller.pid_w.max_output == 10.0

def test_reset(speed_controller: SpeedController) -> None:
    """Test that the reset function clears the internal state."""
    speed_controller.pid_w.update(100.0, 0.0)
    assert speed_controller.pid_w.integral != 0.0
    speed_controller.reset()
    assert speed_controller.pid_w.integral == pytest.approx(0.0)

def test_get_params(speed_controller: SpeedController) -> None:
    """Test retrieval of controller parameters."""
    params = speed_controller.get_params()
    assert params[0] == pytest.approx(0.5)
    assert params[1] == pytest.approx(0.1)

def test_set_params(speed_controller: SpeedController) -> None:
    """Test setting of controller parameters."""
    speed_controller.set_params([1.0, 0.5])
    assert speed_controller.pid_w.kp == pytest.approx(1.0)
    assert speed_controller.pid_w.ki == pytest.approx(0.5)

def test_step(speed_controller: SpeedController) -> None:
    """Test the step function of the speed controller."""
    iq_ref = speed_controller.step(omega_ref=10.0, omega_meas=0.0)
    assert iq_ref == pytest.approx(5.0001)
