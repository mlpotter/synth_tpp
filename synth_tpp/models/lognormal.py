from scipy.stats import lognorm
import numpy as np
from numba import njit
from .base import BaseTPP

def lognormal_renewal_compensators(theta,ℋ_t):
    mean, std = theta
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(variance / mean ** 2 + 1))
    
    t_diff = np.diff(ℋ_t, prepend=0)

    Λs = -np.log(1-lognorm.cdf(t_diff, s=sigma, scale=np.exp(mu)))
    return np.cumsum(Λs)

def lognormal_renewal_intensity(t, ℋ_t, theta):
    mean, std = theta
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(variance / mean ** 2 + 1))

    ℋ_t = ℋ_t[ℋ_t < t]
    if len(ℋ_t) == 0:
        last_event = 0
    else:
        last_event = ℋ_t[-1]

    t_diff = t - last_event

    return lognorm.pdf(t_diff, s=sigma, scale=np.exp(mu)) / lognorm.sf(t_diff, s=sigma, scale=np.exp(mu))

def simulate_lognormal_renewal_N(theta, n_events):
    """
    Simulate a stationary renewal process with log-normal inter-event intervals.
    Intervals are i.i.d. log-normal with given mean and std.
    Simulation stops after n_events events.
    """
    mean, std = theta
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(variance / mean ** 2 + 1))

    events = []
    t = 0.0

    # Convert mean and std to log-normal parameters
    for _ in range(n_events):
        interval = np.random.lognormal(mean=mu, sigma=sigma)
        t += interval
        events.append(t)

    return np.array(events)

def simulate_lognormal_renewal_T(theta,T_max):
    """
    Simulate a stationary renewal process with log-normal inter-event intervals.
    Intervals are i.i.d. log-normal with given mean and std.
    Simulation stops when time exceeds T_max.
    """
    mean, std = theta
    events = []
    t = 0.0

    # Convert mean and std to log-normal parameters
    # mean = exp(mu + sigma^2/2), std^2 = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    variance = std ** 2
    mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
    sigma = np.sqrt(np.log(variance / mean ** 2 + 1))

    while t < T_max:
        interval = np.random.lognormal(mean=mu, sigma=sigma)
        t += interval
        if t < T_max:
            events.append(t)
        else:
            break

    return np.array(events)

class LognormalRenewalTPP(BaseTPP):
    def __init__(self, theta):
        super().__init__(theta)
        
    def intensity(self, t, history):
        return lognormal_renewal_intensity(t, history, self.theta)

    def compensator(self, history):
        return lognormal_renewal_compensators(self.theta, history)

    def generate_T(self, history):
        return simulate_lognormal_renewal_T(self.theta, history)

    def generate_N(self, T, history):
        return simulate_lognormal_renewal_N(self.theta, T)