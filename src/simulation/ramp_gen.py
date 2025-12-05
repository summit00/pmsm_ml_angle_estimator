# simple_ramp.py
import numpy as np

class SimpleRamp:
    """Minimal ramp generator for smooth transitions."""
    
    def __init__(self, ramp_rate=10.0, dt=0.001, start=0.0):
        self.ramp_rate = ramp_rate  # max change per second
        self.dt = dt
        self.current = start
        self.target = start
    
    def set_target(self, target):
        """Set new target value."""
        self.target = target
    
    def update(self):
        """Update and return current ramped value."""
        if self.current == self.target:
            return self.current
        
        step = self.ramp_rate * self.dt
        error = self.target - self.current
        
        if abs(error) <= step:
            self.current = self.target  # reached target
        else:
            self.current += np.sign(error) * step
        
        return self.current