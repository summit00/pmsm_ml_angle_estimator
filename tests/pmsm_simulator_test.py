# src/tests/test_simulator_simple.py

from unittest.mock import MagicMock
import numpy as np
import pytest
from pmsm_simulator import PmsmFocSimulator

# Mock dependencies to isolate the PmsmFocSimulator class for testing
@pytest.fixture
def mock_plant():
    plant = MagicMock()
    plant.p = 2  # Example pole pairs
    plant.ode.return_value = np.zeros(4)
    plant.output.return_value = {"torque": 0.0, "i_d": 0.0, "i_q": 0.0, "theta_m": 0.0, "omega_m": 0.0}
    return plant

@pytest.fixture
def mock_current_controller() -> MagicMock:
    controller = MagicMock()
    controller.step.return_value = (0.0, 0.0) # v_d, v_q
    return controller

@pytest.fixture
def mock_speed_controller() -> MagicMock:
    controller = MagicMock()
    controller.step.return_value = 0.5 # iq_ref
    return controller

def test_simulator_runs_basic_simulation(mock_plant: MagicMock, mock_current_controller: MagicMock, mock_speed_controller: MagicMock) -> None:
    """Test that the simulator can run a short, basic simulation loop and return a pandas DataFrame. """
    # A simple omega reference function that returns a constant speed
    def simple_omega_ref(t) -> float:
        return 100.0

    simulator = PmsmFocSimulator(
        plant=mock_plant,
        current_controller=mock_current_controller,
        speed_controller=mock_speed_controller,
        t_final=0.01,         # Run for a very short time
        omega_ref_func=simple_omega_ref, # Pass the function itself, not the call result
        dt_sim=1e-4           # Use larger dt for fewer steps in test
    )
    # Note: I fixed a bug in your original test where you called simple_omega_ref() immediately.

    # Run the simulation
    df_results = simulator.run()

    # Basic assertions to check if it worked
    assert not df_results.empty
    assert "t" in df_results.columns
    assert "omega_ref" in df_results.columns
    # Check if the speed controller was called at least once
    assert mock_speed_controller.step.called
    # Check if the plant's ODE was called multiple times
    assert mock_plant.ode.call_count > 5


def test_simulator_without_speed_controller_runs(mock_plant: MagicMock, mock_current_controller: MagicMock) -> None:
    """
    Test the simulation runs when the speed controller is explicitly None. 
    This covers the `if self.speed_controller is not None:` branches.
    """
    simulator = PmsmFocSimulator(
        plant=mock_plant,
        current_controller=mock_current_controller,
        speed_controller=None,      # No speed controller
        omega_ref_func=None,        # No reference function needed either
        t_final=0.001,
        dt_sim=1e-4
    )
    
    # It should run without raising an AttributeError
    df_results = simulator.run()
    
    assert not df_results.empty
    assert "i_q_ref" in df_results.columns
    # iq_ref should remain 0.0 throughout the simulation
    assert (df_results["i_q_ref"] == 0.0).all()

