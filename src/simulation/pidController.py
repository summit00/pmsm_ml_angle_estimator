"""PID Controller Module."""


class PIDController:
    """
    A simple PID controller implementation.

    The PID controller computes an output based on the error between a setpoint and a measured value.
    It uses three components:
        - Proportional (P): reacts to the current error
        - Integral (I): reacts to the accumulated error over time
        - Derivative (D): reacts to the rate of change of error

    Equation:
        output = Kp * error + Ki * ∫error dt + Kd * d(error)/dt
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        """
        Initialize the PID controller.

        Parameters:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            dt (float): Time step for updates
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0
        self.min_output = -float("inf")
        self.max_output = float("inf")

    def reset(self) -> None:
        """
        Reset the internal state of the controller (integral and previous error).
        """
        self.integral = 0.0
        self.prev_error = 0.0

    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """
        Set saturation limits for the controller output.

        Parameters:
            min_output (float): Minimum output value
            max_output (float): Maximum output value
        """
        self.min_output = min_output
        self.max_output = max_output

    def update(self, setpoint: float, measurement: float) -> float:
        """
        Compute the PID controller output.

        Parameters:
            setpoint (float): Desired target value
            measurement (float): Current measured value

        Returns:
            float: Controller output
        """
        error = setpoint - measurement

        # Integral term
        self.integral += error * self.dt

        # Derivative term
        derivative = (error - self.prev_error) / self.dt

        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply saturation limits
        output = max(self.min_output, min(self.max_output, output))

        # Save error for next derivative calculation
        self.prev_error = error

        return output
