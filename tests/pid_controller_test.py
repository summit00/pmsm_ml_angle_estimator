"""Unit tests for the PIDController class using pytest framework."""

import pytest
from pid_controller import PIDController

# Use pytest fixtures to create a fresh controller instance for each test
@pytest.fixture
def pid_controller() -> PIDController:
    """Fixture that provides a default PID controller instance."""
    return PIDController(kp=0.5, ki=0.1, kd=0.0, dt=0.1)

def test_initialization(pid_controller: PIDController) -> None:
    """Test that the controller initializes with the correct parameters."""
    assert pid_controller.kp == 0.5
    assert pid_controller.ki == 0.1
    assert pid_controller.kd == 0.0
    assert pid_controller.dt == 0.1
    assert pid_controller.integral == 0.0
    assert pid_controller.prev_error == 0.0

def test_proportional_response(pid_controller: PIDController) -> None:
    """Test that the proportional term works correctly."""
    output: float = pid_controller.update(setpoint=10.0, measurement=0.0)
    assert output == pytest.approx(5.1)


def test_integral_buildup() -> None:
    """Test that the integral term accumulates error over time."""
    controller: PIDController = PIDController(kp=0.0, ki=1.0, kd=0.0, dt=0.5)
    output_1: float = controller.update(setpoint=10.0, measurement=0.0)
    assert output_1 == pytest.approx(5.0)
    output_2: float = controller.update(setpoint=10.0, measurement=0.0)
    assert output_2 == pytest.approx(10.0)


def test_derivative_response() -> None:
    """Test that the derivative term reacts to the rate of change of error."""
    controller: PIDController = PIDController(kp=0.0, ki=0.0, kd=2.0, dt=0.1)
    controller.update(setpoint=10.0, measurement=0.0)
    output: float = controller.update(setpoint=10.0, measurement=1.0)
    assert output == pytest.approx(-20.0)


def test_output_limits(pid_controller: PIDController) -> None:
    """Test that the output saturation limits work correctly."""
    pid_controller.set_output_limits(min_output=0.0, max_output=10.0)
    output: float = pid_controller.update(setpoint=1000.0, measurement=0.0)
    assert output == pytest.approx(10.0)
    output_low: float = pid_controller.update(setpoint=-1000.0, measurement=0.0)
    assert output_low == pytest.approx(0.0)


def test_reset_functionality(pid_controller: PIDController) -> None:
    """Test that resetting clears the internal state."""
    pid_controller.update(setpoint=10.0, measurement=0.0)
    assert pid_controller.integral != 0.0
    assert pid_controller.prev_error != 0.0
    pid_controller.reset()
    assert pid_controller.integral == pytest.approx(0.0)
    assert pid_controller.prev_error == pytest.approx(0.0)
