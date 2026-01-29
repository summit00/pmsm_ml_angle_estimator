"""Unit tests for the transform class using pytest framework."""

import pytest
from transform import ClarkeParkTransform

def test_clarke_transform() -> None:
    """Test Clarke Transform functionality."""
    transform = ClarkeParkTransform()
    a, b = 1.0, 3.0
    alpha, beta = transform.clarke_transform(a, b)
    assert alpha == pytest.approx(1.0)
    assert beta == pytest.approx(4.041451, rel=1e-5)

def test_inverse_clarke_transform() -> None:
    """Test Inverse Clarke Transform functionality."""
    transform = ClarkeParkTransform()
    alpha, beta = 1.0, 3.0
    a, b, c = transform.inverse_clarke_transform(alpha, beta)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(2.098076, rel=1e-5)
    assert c == pytest.approx(-3.098076, rel=1e-5)

def test_park_transform() -> None:
    """Test Park Transform functionality."""
    transform = ClarkeParkTransform()
    alpha, beta, theta = 1.0, 3.0, 1.5
    d, q = transform.park_transform(alpha, beta, theta)
    assert d == pytest.approx(3.063222, rel=1e-5)
    assert q == pytest.approx(-0.785283, rel=1e-5)

def test_inverse_transform() -> None:
    """Test Park Transform with zero angle."""
    transform = ClarkeParkTransform()
    alpha, beta, theta = 1.0, 3.0, 1.5
    d, q = transform.inverse_park_transform(alpha, beta, theta)
    assert d == pytest.approx(-2.921748, rel=1e-5)
    assert q == pytest.approx(1.209707, rel=1e-5)
