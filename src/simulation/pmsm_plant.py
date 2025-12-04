# pmsm_plant.py
import numpy as np


class PmsmPlant:
    """
    Permanent Magnet Synchronous Motor (PMSM) plant model.
    
    Implements the dynamic equations for a PMSM in the rotor reference frame (dq-frame)
    including both electrical and mechanical dynamics.
    
    Attributes:
        Rs (float): Stator resistance [Ω]
        Ld (float): d-axis inductance [H]
        Lq (float): q-axis inductance [H]
        psi_f (float): Permanent magnet flux linkage [Wb]
        p (int): Number of pole pairs
        J (float): Rotor inertia [kg·m²]
        B (float): Viscous friction coefficient [N·m·s]
        load_torque_func (callable): Function returning load torque as a function of time
    """

    def __init__(
        self,
        Rs=0.1,       
        Ld=1e-3,      
        Lq=1.2e-3,    
        psi_f=0.05,   
        p=3,          
        J=0.01,       
        B=0.001,      
        load_torque_func=None, 
    ):
        # Electrical parameters
        self.Rs = Rs
        self.Ld = Ld
        self.Lq = Lq
        self.psi_f = psi_f
        self.p = p

        # Mechanical parameters
        self.J = J
        self.B = B

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
            x: State vector [i_d, i_q, omega_m]
            u: Input vector [v_d, v_q]
            
        Returns:
            dx/dt: Derivative of state vector
        """
        i_d, i_q, omega_m= x
        v_d, v_q = u

        # electrical speed [rad/s]
        omega_e = self.p * omega_m

        # electrical dynamics
        di_d_dt = (v_d - self.Rs * i_d + omega_e * self.Lq * i_q) / self.Ld
        di_q_dt = (v_q - self.Rs * i_q - omega_e * self.Ld * i_d - omega_e * self.psi_f) / self.Lq

        # electromagnetic torque
        torque_e = 1.5 * self.p * (self.psi_f * i_q + (self.Ld - self.Lq) * i_d * i_q)

        # mechanical dynamics
        torque_load = self.load_torque_func(t)
        domega_m_dt = (torque_e - torque_load - self.B * omega_m) / self.J

        # Return state derivatives
        return np.array([di_d_dt, di_q_dt, domega_m_dt])

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
        i_d, i_q, omega_m = x
        
        # Calculate derived quantities
        omega_e = self.p * omega_m  # Electrical speed [rad/s]
        
        # Electromagnetic torque [N·m]
        torque_e = 1.5 * self.p * (
            self.psi_f * i_q + (self.Ld - self.Lq) * i_d * i_q
        )
        
        # Load torque [N·m]
        torque_load = self.load_torque_func(t)
        
        # Return comprehensive output dictionary
        return {
            "i_d": i_d,           # d-axis current [A]
            "i_q": i_q,           # q-axis current [A]
            "omega_m": omega_m,   # Mechanical speed [rad/s]
            "omega_e": omega_e,   # Electrical speed [rad/s]
            "torque_e": torque_e, # Electromagnetic torque [N·m]
            "torque_load": torque_load, # Load torque [N·m]
        }
