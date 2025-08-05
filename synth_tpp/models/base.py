import numpy as np
from abc import ABC, abstractmethod

class BaseTPP(ABC):
    def __init__(self, theta=None):
        """
        Initialize the base temporal point process model with given parameter theta.

        Args:
            theta: Model parameter(s).
        """
        self.theta = theta
    @abstractmethod
    def intensity(self, t, history):
        """Compute the intensity function at time t given history."""
        pass

    @abstractmethod
    def compensator(self, t, history):
        """Compute the compensator up to time t given history."""
        pass

    @abstractmethod
    def generate_T(self, history):
        """Generate the next event time given history."""
        pass

    @abstractmethod
    def generate_N(self, T, history):
        """Generate the number of events up to time T given history."""
        pass