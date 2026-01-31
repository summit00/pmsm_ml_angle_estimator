import pytest
from ramp_gen import SimpleRamp


@pytest.fixture
def ramp() -> SimpleRamp:
    """Provides a ramp moving at 10 units/sec with 1ms steps."""
    return SimpleRamp(ramp_rate=10.0, dt=0.001, start=0.0)

def test_initialization(ramp: SimpleRamp) -> None:
    """Test that the ramp generator initializes with the correct parameters."""
    assert ramp.ramp_rate == 10.0
    assert ramp.dt == 0.001
    assert ramp.current == 0.0
    assert ramp.target == 0.0

def test_ramp_up(ramp: SimpleRamp) -> None:
    """Test that the ramp increases by exactly step size."""
    ramp.set_target(5.0)
    # First update: 0.0 + (10.0 * 0.001) = 0.01
    val = ramp.update()
    assert val == pytest.approx(0.01)
    # Second update: 0.01 + 0.01 = 0.02
    val = ramp.update()
    assert val == pytest.approx(0.02)

def test_reaches_target_exactly(ramp: SimpleRamp) -> None:
    """Test that it doesn't overshoot and stops at target."""
    # Move towards a very small target
    ramp.set_target(0.015)
    
    ramp.update()  # Now at 0.01
    val = ramp.update()  # Step is 0.01, but error is only 0.005. Should land on 0.015.
    
    assert val == 0.015
    assert ramp.current == 0.015
    
    # Further updates should stay at target
    assert ramp.update() == 0.015

def test_ramp_down(ramp: SimpleRamp) -> None:
    """Test that it handles negative transitions."""
    ramp.current = 1.0
    ramp.set_target(0.0)
    
    val = ramp.update()
    assert val == pytest.approx(0.99) # 1.0 - 0.01

def test_timing_accuracy(ramp: SimpleRamp) -> None:
    """Test if it reaches target in the mathematically expected time."""
    # To go from 0 to 1 at 10 units/sec should take 0.1 seconds.
    # 0.1s / 0.001dt = 100 steps.
    ramp.set_target(1.0)
    
    for _ in range(100):
        val = ramp.update()
        
    assert val == pytest.approx(1.0)