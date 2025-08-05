from hawkesbook import exp_hawkes_compensators
import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from .base import BaseTPP

def simulate_self_correcting_T(T_max):
    t = 0
    events = []

    while t < T_max:
        u = np.random.uniform()
        k = len(events)

        t = np.log(-np.log(u)*np.exp(k) + np.exp(t))

        if t < T_max:
            events.append(t)
        else:
            break
        
    return np.array(events)

def simulate_self_correcting_N(n_points):
    """
    Simulate a self-correcting point process with intensity λ(t | H_t) = exp(t - N_t)
    until exactly `n_points` events are generated.
    """
    t = 0
    events = []

    while len(events) < n_points:
        N_t = len(events)
        u = np.random.uniform()
        t = np.log(-np.log(u)*np.exp(N_t) + np.exp(t))

        events.append(t)

    return np.array(events)

def self_correcting_compensators(ℋ_t):
    N_ts = np.arange(len(ℋ_t))
    Λ = 0
    Λs = np.empty(len(ℋ_t), dtype=np.float64)
    for i, N_t in enumerate(N_ts):
        Λ += np.exp(-N_t) * (np.exp(ℋ_t[i]) - np.exp(ℋ_t[i-1]) if i > 0 else np.exp(ℋ_t[i]))
        Λs[i] = Λ

    return Λs

def self_correcting_intensity(t,ℋ_t):
    ℋ_t = ℋ_t[ℋ_t < t]
    N_t = len(ℋ_t)

    return np.exp(t - N_t)

class SelfCorrectingTPP(BaseTPP):
    def intensity(self, t, history):
        return self_correcting_intensity(t, history)

    def compensator(self, history):
        return self_correcting_compensators(history)

    def generate_T(self, T_max):
        return simulate_self_correcting_T(T_max)

    def generate_N(self, n_points):
        return simulate_self_correcting_N(n_points)
