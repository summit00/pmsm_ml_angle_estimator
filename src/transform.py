""""Clarke and Park Transformations for three-phase systems."""
from math import cos, sin, sqrt


class ClarkeParkTransform:
    """Clarke/Park Transform Class for three-phase system analysis."""

    def __init__(self) -> None:
        """Initialize Clarke/Park Transform class."""

    def clarke_transform(self, a: float, b: float) -> tuple[float, float]:
        """Clarke Transform (3-phase to αβ stationary frame).

        Args:
            a: Phase A quantity.
            b: Phase B quantity.

        Returns:
            tuple: (alpha, beta) components in stationary frame
        """
        alpha = a
        beta = (a + 2 * b) / sqrt(3)

        return alpha, beta

    def inverse_clarke_transform(self, alpha: float, beta: float) -> tuple[float, float, float]:
        """Inverse Clarke Transform (αβ stationary frame to 3-phase).

        Args:
            alpha: Alpha component in stationary frame.
            beta: Beta component in stationary frame.
            balanced (bool): If True, returns balanced three-phase quantities

        Returns:
            tuple: (a, b, c) three-phase quantities
        """
        a = alpha
        b = -0.5 * alpha + (sqrt(3) / 2) * beta
        c = -0.5 * alpha - (sqrt(3) / 2) * beta

        return a, b, c

    def park_transform(self, alpha: float, beta: float, theta: float) -> tuple[float, float]:
        """Park Transform (αβ stationary frame to dq rotating frame).

        Args:
            alpha: Alpha component in stationary frame.
            beta: Beta component in stationary frame.
            theta: Electrical angle in radians.

        Returns:
            tuple: (d, q) components in rotating frame.
        """
        # Transformation matrix
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        d = alpha * cos_theta + beta * sin_theta
        q = -alpha * sin_theta + beta * cos_theta

        return d, q

    def inverse_park_transform(self, d: float, q: float, theta: float) -> tuple[float, float]:
        """Inverse Park Transform (dq rotating frame to αβ stationary frame).

        Args:
            d: D-axis component in rotating frame.
            q: Q-axis component in rotating frame.
            theta: Electrical angle in radians.

        Returns:
            tuple: (alpha, beta) components in stationary frame.
        """
        # Inverse transformation matrix
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        alpha = d * cos_theta - q * sin_theta
        beta = d * sin_theta + q * cos_theta

        return alpha, beta
