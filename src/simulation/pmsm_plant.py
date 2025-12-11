"""PMSM plant model implementation."""

import numpy as np


class PmsmPlant:
    """
    Permanent Magnet Synchronous Motor (PMSM) plant model.

    Implements the dynamic equations for a PMSM in the rotor reference frame (dq-frame)
    including both electrical and mechanical dynamics.

    Attributes:
        r_s (float): Stator resistance [Ω]
        l_d (float): d-axis inductance [H]
        l_q (float): q-axis inductance [H]
        psi_f (float): Permanent magnet flux linkage [Wb]
        p (int): Number of pole pairs
        j (float): Rotor inertia [kg·m²]
        b (float): Viscous friction coefficient [N·m·s]
        load_torque_func (callable): Function returning load torque as a function of time
    """

    def __init__(
        self,
        r_s=0.1,
        l_d=1e-3,
        l_q=1.2e-3,
        psi_f=0.05,
        p=3,
        j=0.01,
        b=0.001,
        load_torque_func=None,
    ):
        # Electrical parameter_s
        self.r_s = r_s
        self.l_d = l_d
        self.l_q = l_q
        self.psi_f = psi_f
        self.p = p

        # Mechanical parameters
        self.j = j
        self.b = b

        # Load torque function (default: zero torque).
        if load_torque_func is None:
            self.load_torque_func = lambda t: 0.0
        else:
            self.load_torque_func = load_torque_func

    def ode(self, t, x, u):
        """
        Compute the state derivatives.

        The ODEs describe the dynamic behavior of the PMSM:
        1. Electrical dynamics (dq-frame)
        2. Mechanical dynamics

        Args:
            t: Current time [s]
            x: State vector [i_d, i_q, theta_m, omega_m]
            u: Input vector [v_d, v_q]

        Returns:
            dx/dt: Derivative of state vector
        """
        i_d, i_q, theta_m, omega_m = x
        v_d, v_q = u

        # electrical speed [rad/s]
        omega_e = self.p * omega_m

        # electrical dynamics
        di_d_dt = (v_d - self.r_s * i_d + omega_e * self.l_q * i_q) / self.l_d
        di_q_dt = (
            v_q - self.r_s * i_q - omega_e * self.l_d * i_d - omega_e * self.psi_f
        ) / self.l_q

        dtheta_m_dt = omega_m

        # electromagnetic torque
        torque_e = 1.5 * self.p * (self.psi_f * i_q + (self.l_d - self.l_q) * i_d * i_q)

        # mechanical dynamics
        torque_load = self.load_torque_func(t)
        domega_m_dt = (torque_e - torque_load - self.b * omega_m) / self.j

        # Return state derivatives
        return np.array([di_d_dt, di_q_dt, dtheta_m_dt, domega_m_dt])

    def output(self, t, x):
        """
        Compute and return all relevant outputs for logging and analysis.

        Args:
            t: Current time [s]
            x: State vector [i_d, i_q, ω_m]

        Returns:
            Dictionary containing all calculated outputs
        """
        # Unpack states
        i_d, i_q, theta_m, omega_m = x

        # Angle rolling.
        theta_e_raw = self.p * theta_m

        # Wrap angle to [0, 2π] for easier interpretation
        theta_wrapped = theta_m % (2 * np.pi)
        theta_e = (theta_wrapped * self.p) % (2 * np.pi)

        # Calculate derived quantities
        omega_e = self.p * omega_m  # Electrical speed [rad/s]

        # Electromagnetic torque [N·m]
        torque_e = 1.5 * self.p * (self.psi_f * i_q + (self.l_d - self.l_q) * i_d * i_q)

        # Load torque [N·m]
        torque_load = self.load_torque_func(t)

        # Return comprehensive output dictionary
        return {
            "i_d": i_d,  # d-axis current [A]
            "i_q": i_q,  # q-axis current [A]
            "theta_m": theta_wrapped,  # Mechanical angle [rad], wrapped 0-2π
            "theta_e": theta_e,  # Electrical angle [rad], wrapped 0-2π
            "theta_m_raw": theta_m,  # Mechanical angle [rad], raw
            "theta_e_raw": theta_e_raw,  # Electrical angle [rad], raw
            "omega_m": omega_m,  # Mechanical speed [rad/s]
            "omega_e": omega_e,  # Electrical speed [rad/s]
            "torque_e": torque_e,  # Electromagnetic torque [N·m]
            "torque_load": torque_load,  # Load torque [N·m]
        }
