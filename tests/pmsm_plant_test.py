import pytest
from pmsm_plant import PmsmPlant
import numpy as np

@pytest.fixture
def pmsm_plant() -> PmsmPlant:
    """Fixture that provides a default PMSM plant instance."""
    return PmsmPlant(
        r_s=0.1,
        l_d=1e-3,
        l_q=1.2e-3,
        psi_f=0.05,
        p=3,
        j=0.01,
        b=0.001,
    )

def test_pmsm_plant_initialization(pmsm_plant: PmsmPlant) -> None:
    """Test that the PMSM plant initializes with correct parameters."""
    assert pmsm_plant.r_s == 0.1
    assert pmsm_plant.l_d == 1e-3
    assert pmsm_plant.l_q == 1.2e-3
    assert pmsm_plant.psi_f == 0.05
    assert pmsm_plant.p == 3
    assert pmsm_plant.j == 0.01
    assert pmsm_plant.b == 0.001

def test_pmsm_plant_ode(pmsm_plant: PmsmPlant) -> None:
    """Test the ODE computation of the PMSM plant."""
    t = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0])  # [i_d, i_q, theta_m, omega_m]
    u = np.array([0.0, 0.0])            # [v_d, v_q]

    dxdt = pmsm_plant.ode(t, x, u)

    # Check that the output is a numpy array of correct shape
    assert isinstance(dxdt, np.ndarray)
    assert dxdt.shape == (4,)

    # Check that the derivatives are computed correctly for zero inputs
    assert dxdt[0] == pytest.approx(0.0)  # di_d/dt
    assert dxdt[1] == pytest.approx(0.0)  # di_q/dt
    assert dxdt[2] == pytest.approx(0.0)  # dtheta_m/dt
    assert dxdt[3] == pytest.approx(0.0)  # domega_m/dt

def test_pmsm_plant_output(pmsm_plant: PmsmPlant) -> None:
    """Test the output computation of the PMSM plant at non-zero state."""
    t = 0.5
    # Example state: some current, angle, and speed
    x = np.array([10.0, 20.0, np.pi / 4, 100.0])  # [i_d, i_q, theta_m, omega_m]

    outputs = pmsm_plant.output(t, x)

    # Check that expected keys are in the output dictionary
    expected_keys = [
        "i_d", "i_q", "theta_m", "theta_e", "theta_m_raw", "theta_e_raw",
        "omega_m", "omega_e", "torque_e", "torque_load"
    ]
    assert all(key in outputs for key in expected_keys)

    # Verify some computed values with approx checks
    # omega_e = p * omega_m = 3 * 100.0 = 300.0
    assert outputs["omega_e"] == pytest.approx(300.0)
    # Check that angles are wrapped correctly (e.g. theta_m should be wrapped version of pi/4)
    assert outputs["theta_m"] == pytest.approx(np.pi / 4)
    # theta_e = (p * theta_m) % (2 * pi)
    expected_theta_e = (3 * (np.pi / 4)) % (2 * np.pi)
    assert outputs["theta_e"] == pytest.approx(expected_theta_e)

    # Check torque calculation (simple non-zero check as exact value is complex)
    assert outputs["torque_e"] != pytest.approx(0.0)
    assert outputs["torque_load"] == pytest.approx(0.0) # Using default zero-torque func


def test_pmsm_plant_with_custom_load_torque_func() -> None:
    """Test the initialization and ODE with a non-default load torque function."""
    # Define a simple custom load function: torque increases with time
    def custom_load(t: float) -> float:
        return 0.1 * t
    plant = PmsmPlant(load_torque_func=custom_load, j=0.01, b=0.001)

    t = 1.0
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0])

    dxdt = plant.ode(t, x, u)
    outputs = plant.output(t, x)

    # Verify the custom load function was used in the ODE calculation
    # For zero currents, torque_e is 0. domega_m/dt = (0 - torque_load - 0) / J
    expected_torque_load = custom_load(t) # 0.1 * 1.0 = 0.1
    expected_domega_dt = -expected_torque_load / plant.j # -0.1 / 0.01 = -10.0

    assert outputs["torque_load"] == pytest.approx(expected_torque_load)
    assert dxdt[3] == pytest.approx(expected_domega_dt)
