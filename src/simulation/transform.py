from math import sin, cos, sqrt


class ClarkeParkTransform:
    """
    Clarke/Park Transform Class for three-phase system analysis.
    Clarke transform converts 3-phase quantities to 2-phase stationary reference frame (αβ).
    Park transform converts stationary frame (αβ) to rotating reference frame (dq).
    """

    def __init__(self):
        """Initialize Clarke/Park Transform class."""
        pass

    def clarke_transform(self, a, b, c, balanced=True):
        """
        Clarke Transform (3-phase to αβ stationary frame)

        Args:
            a, b, c: Three-phase quantities (voltages, currents, etc.)
            balanced (bool): If True, uses balanced transformation (c = -a-b)
                           If False, assumes abc are independent

        Returns:
            tuple: (alpha, beta) components in stationary frame
        """
        if balanced:
            # For balanced three-phase systems: a + b + c = 0
            # So we can compute c = -(a + b) if needed, or use the standard transform
            alpha = a
            beta = (a + 2 * b) / sqrt(3)  # Alternative formulation
            # alpha = (2/3) * (a - 0.5*b - 0.5*c)  # Another common formulation
            # beta = (2/3) * (sqrt(3)/2 * (b - c))
        else:
            # Standard Clarke transform
            alpha = (2 / 3) * (a - 0.5 * b - 0.5 * c)
            beta = (2 / 3) * (sqrt(3) / 2 * (b - c))

        return alpha, beta

    def inverse_clarke_transform(self, alpha, beta, balanced=True):
        """
        Inverse Clarke Transform (αβ stationary frame to 3-phase)

        Args:
            alpha, beta: Stationary frame components
            balanced (bool): If True, returns balanced three-phase quantities

        Returns:
            tuple: (a, b, c) three-phase quantities
        """
        if balanced:
            a = alpha
            b = -0.5 * alpha + (sqrt(3) / 2) * beta
            c = -0.5 * alpha - (sqrt(3) / 2) * beta
        else:
            # Standard inverse Clarke transform
            a = alpha
            b = -0.5 * alpha + (sqrt(3) / 2) * beta
            c = -0.5 * alpha - (sqrt(3) / 2) * beta

        return a, b, c

    def park_transform(self, alpha, beta, theta):
        """
        Park Transform (αβ stationary frame to dq rotating frame)

        Args:
            alpha, beta: Stationary frame components
            theta: Electrical angle in radians

        Returns:
            tuple: (d, q) components in rotating frame
        """
        # Transformation matrix
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        d = alpha * cos_theta + beta * sin_theta
        q = -alpha * sin_theta + beta * cos_theta

        return d, q

    def inverse_park_transform(self, d, q, theta):
        """
        Inverse Park Transform (dq rotating frame to αβ stationary frame)

        Args:
            d, q: Rotating frame components
            theta: Electrical angle in radians

        Returns:
            tuple: (alpha, beta) components in stationary frame
        """
        # Inverse transformation matrix
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        alpha = d * cos_theta - q * sin_theta
        beta = d * sin_theta + q * cos_theta

        return alpha, beta

    def transform_abc_to_dq(self, a, b, c, theta, balanced=True):
        """
        Combined transformation from 3-phase to dq rotating frame

        Args:
            a, b, c: Three-phase quantities
            theta: Electrical angle in radians
            balanced (bool): If True, assumes balanced three-phase system

        Returns:
            tuple: (d, q) components in rotating frame
        """
        # Step 1: Clarke transform
        alpha, beta = self.clarke_transform(a, b, c, balanced)

        # Step 2: Park transform
        d, q = self.park_transform(alpha, beta, theta)

        return d, q

    def transform_dq_to_abc(self, d, q, theta, balanced=True):
        """
        Combined transformation from dq rotating frame to 3-phase

        Args:
            d, q: Rotating frame components
            theta: Electrical angle in radians
            balanced (bool): If True, returns balanced three-phase quantities

        Returns:
            tuple: (a, b, c) three-phase quantities
        """
        # Step 1: Inverse Park transform
        alpha, beta = self.inverse_park_transform(d, q, theta)

        # Step 2: Inverse Clarke transform
        a, b, c = self.inverse_clarke_transform(alpha, beta, balanced)

        return a, b, c
