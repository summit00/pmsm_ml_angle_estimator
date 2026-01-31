""""Unit tests for the FocCurrentController class."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from foc_controller import FocCurrentController


# We patch the *class* globally, but configure the instances within tests
@pytest.fixture(autouse=True)
def mock_pid_controller_class() -> MagicMock:
    """Global patch of the PIDController class using autouse to ensure it's always active."""
    with patch('foc_controller.PIDController') as MockPID:
        yield MockPID


# --- Initialization and Parameter Tests ---

def test_controller_initialization(mock_pid_controller_class: MagicMock) -> None:
    """Verify initialization sets up parameters and PID controllers correctly."""
    ctrl = FocCurrentController(dt=0.001)
    assert ctrl.dt == 0.001
    assert ctrl.v_limit == 50.0

    # Verify that PID controllers were instantiated twice and configured
    assert mock_pid_controller_class.call_count == 2
    # Ensure set_output_limits was called on the mock instances
    assert mock_pid_controller_class.return_value.set_output_limits.call_count == 2



def test_get_and_set_params() -> None:
    """Verify that parameters can be retrieved and updated correctly using distinct mocks."""
    with patch('foc_controller.PIDController') as MockPIDClass:
        # Create two distinct mock instances manually
        mock_d = MagicMock()
        mock_q = MagicMock()
        # Configure the patched class to return these two distinct mocks in order
        MockPIDClass.side_effect = [mock_d, mock_q]

        # Initialize the controller; it now gets distinct mocks for pid_d and pid_q
        ctrl = FocCurrentController(dt=0.001)

        # Manually set the initial default values on the *distinct* mock instances
        mock_d.kp = 5.0
        mock_d.ki = 1000.0
        mock_q.kp = 5.0
        mock_q.ki = 1000.0
        default_gains = np.array([5.0, 1000.0, 5.0, 1000.0])
        # Now this assertion should pass because they have separate state
        np.testing.assert_array_almost_equal(ctrl.get_params(), default_gains)

        # --- Test set_params ---
        new_gains = np.array([10.0, 2000.0, 12.0, 2500.0])
        ctrl.set_params(new_gains)
        # Verify that the attributes of the individual mock objects were updated by set_params
        assert mock_d.kp == 10.0
        assert mock_d.ki == 2000.0
        assert mock_q.kp == 12.0
        assert mock_q.ki == 2500.0

        # Verify that get_params returns these newly set values
        retrieved_gains = ctrl.get_params()
        np.testing.assert_array_almost_equal(retrieved_gains, new_gains)



def test_reset_functionality(mock_pid_controller_class: MagicMock) -> None:
    """Verify that the reset method calls reset on internal PID instances."""
    ctrl = FocCurrentController(dt=0.001)
    ctrl.reset()
    # The return_value is the shared mock instance that gets used for both pid_d and pid_q
    assert mock_pid_controller_class.return_value.reset.call_count == 2


# --- Step Logic Tests ---

def test_step_zero_speed_no_decoupling(mock_pid_controller_class: MagicMock) -> None:
    """Verify step works with zero speed (no decoupling terms)."""
    ctrl = FocCurrentController(dt=0.001)
    # Configure the shared mock instance to return specific values
    mock_pid_controller_class.return_value.update.side_effect = [1.0, 2.0] # Return 1.0 for d, 2.0 for q

    v_d, v_q = ctrl.step(i_d_ref=1.0, i_q_ref=5.0, i_d_meas=0.0, i_q_meas=0.0, omega_e=0.0)

    assert v_d == pytest.approx(1.0)
    assert v_q == pytest.approx(2.0)
    # Ensure update was called on the mock instance (twice: once for d, once for q)
    assert mock_pid_controller_class.return_value.update.call_count == 2


def test_step_decoupling_terms(mock_pid_controller_class: MagicMock) -> None:
    """Verify that decoupling feedforward terms are calculated and applied."""
    ctrl = FocCurrentController(dt=0.001)
    omega_e = 100.0
    i_d_meas = 5.0
    i_q_meas = 10.0

    # Ensure PIDs return 0 to isolate the decoupling terms
    mock_pid_controller_class.return_value.update.side_effect = [0.0, 0.0]
    v_d, v_q = ctrl.step(0.0, 0.0, i_d_meas, i_q_meas, omega_e)

    expected_v_d = -1.2 # -100 * 1.2e-3 * 10 
    expected_v_q = 5.5  # +100 * 1e-3 * 5 + 100 * 0.05
    assert v_d == pytest.approx(expected_v_d)
    assert v_q == pytest.approx(expected_v_q)


def test_step_voltage_limiting(mock_pid_controller_class: MagicMock) -> None:
    """Verify that the voltage magnitude limit branch is hit and scaling occurs."""
    ctrl = FocCurrentController(dt=0.001, v_limit=10.0)
    # Configure PIDs to return large values that *exceed* the limit
    # e.g. magnitude will be sqrt(100^2 + 0^2) = 100V
    mock_pid_controller_class.return_value.update.side_effect = [100.0, 0.0] 

    v_d, v_q = ctrl.step(0.0, 0.0, 0.0, 0.0, omega_e=0.0)
    v_mag_actual = np.sqrt(v_d**2 + v_q**2)
    # The resulting magnitude should be clipped to the v_limit (10.0 V)
    assert v_mag_actual == pytest.approx(ctrl.v_limit)
    assert v_d == pytest.approx(10.0)
    assert v_q == pytest.approx(0.0)
