from hawkesbook import exp_hawkes_compensators
import numpy as np
from numba import njit
import numpy.random as rnd

from abc import ABC, abstractmethod
from .base import BaseTPP

def simulate_self_correcting_T(theta,T_max):
    t = 0
    events = []
    alpha,beta = theta
    while t < T_max:
        u = np.random.uniform()
        k = len(events)

        t = np.log(-alpha * np.log(u)*np.exp(beta*k) + np.exp(alpha*t)) / alpha 

        if t < T_max:
            events.append(t)
        else:
            break
        
    return np.array(events)

def simulate_self_correcting_N(theta,n_points):
    """
    Simulate a self-correcting point process with intensity λ(t | H_t) = exp(t - N_t)
    until exactly `n_points` events are generated.
    """
    t = 0
    events = []
    alpha, beta = theta
    while len(events) < n_points:
        N_t = len(events)
        u = np.random.uniform()
        t = np.log(-alpha*np.log(u)*np.exp(beta*N_t) + np.exp(alpha*t)) / alpha

        events.append(t)

    return np.array(events)

def self_correcting_compensators(theta,ℋ_t):
    alpha, beta = theta
    N_ts = np.arange(len(ℋ_t))
    Λ = 0
    Λs = np.empty(len(ℋ_t), dtype=np.float64)
    for i, N_t in enumerate(N_ts):
        Λ += np.exp(-beta*N_t) * (np.exp(alpha*ℋ_t[i]) - np.exp(alpha*ℋ_t[i-1]) if i > 0 else np.exp(alpha * ℋ_t[i]))
        Λs[i] = Λ

    return Λs

def self_correcting_intensity(t,ℋ_t,theta):
    alpha, beta = theta
    ℋ_t = ℋ_t[ℋ_t < t]
    N_t = len(ℋ_t)

    return np.exp(alpha*t - beta*N_t)

class SelfCorrectingTPP(BaseTPP):
    def __init__(self,theta=np.array([1.0, 1.0])):
        """
        """
        super().__init__(theta)

    def intensity(self, t, history):
        return self_correcting_intensity(t, history, self.theta)

    def compensator(self, history):
        return self_correcting_compensators(self.theta, history)

    def generate_T(self, T_max):
        return simulate_self_correcting_T(self.theta, T_max)

    def generate_N(self, n_points):
        return simulate_self_correcting_N(self.theta, n_points)
