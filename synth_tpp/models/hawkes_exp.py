from hawkesbook import exp_hawkes_compensators
import numpy as np
from numba import njit
from synth_tpp.models.base import BaseTPP
import numpy.random as rnd

# @njit(nogil=True)
def hawkes_exp_simulate_by_composition_T(𝛉, T):
    λ, α, β = 𝛉
    λˣ_k = λ
    t_k = 0

    ℋ = []
    while t_k < T:
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        # Technically the following works, but without @njit
        # it will print out "RuntimeWarning: invalid value encountered in log".
        # This is because 1 + β/(λˣ_k + α - λ)*np.log(U_2) can be negative
        # so T_2 can be np.NaN. The Dassios & Zhao (2013) algorithm checks if this
        # expression is negative and handles it separately, though the lines
        # below have the same behaviour as t_k = min(T_1, np.NaN) will be T_1. 
        
        # simulate the (k+1)th interarrival-time 
        delta_T_1 = -np.log(U_1) / λ
        delta_T_2 = -np.log(1 + β/max(λˣ_k - λ,1e-32)*np.log(U_2))/β
        delta_T = min(delta_T_1, delta_T_2)

        t_k = t_k + delta_T

        ℋ.append(t_k)

        # record the change at the jump-time t_{k+1}
        λˣ_k = λ + (λˣ_k - λ) * np.exp(-β * delta_T) + α
        
    return ℋ[:-1]

def hawkes_exp_simulate_by_composition_N(𝛉, N):
    λ, α, β = 𝛉
    λˣ_k = λ
    t_k = 0

    ℋ = np.empty(N, dtype=np.float64)
    for k in range(N):
        U_1 = rnd.rand()
        U_2 = rnd.rand()

        # Technically the following works, but without @njit
        # it will print out "RuntimeWarning: invalid value encountered in log".
        # This is because 1 + β/(λˣ_k + α - λ)*np.log(U_2) can be negative
        # so T_2 can be np.NaN. The Dassios & Zhao (2013) algorithm checks if this
        # expression is negative and handles it separately, though the lines
        # below have the same behaviour as t_k = min(T_1, np.NaN) will be T_1. 

        # simulate the (k+1)th interarrival-time 
        delta_T_1 = -np.log(U_1) / λ
        delta_T_2 = -np.log(1 + β/max(λˣ_k - λ,1e-32)*np.log(U_2))/β
        delta_T = min(delta_T_1, delta_T_2)

        t_k = t_k + delta_T

        ℋ[k] = t_k

        # record the change at the jump-time t_{k+1}
        λˣ_k = λ + (λˣ_k - λ) * np.exp(-β * delta_T) + α
          
    return ℋ

def hawkes_exp_intensity(t, ℋ_t, 𝛉):
    λ, α, β = 𝛉
    λˣ = λ
    ℋ_t = np.array(ℋ_t)
    ℋ_t = ℋ_t[ℋ_t < t]  # Filter events before time t
    for t_i in ℋ_t:
        λˣ += α * np.exp(-β * (t - t_i))
    return λˣ


class HawkesExpTPP(BaseTPP):
    def __init__(self, theta):
        super().__init__(theta)

    def generate_T(self, T):
        return hawkes_exp_simulate_by_composition_T(self.theta, T)

    def generate_N(self, N):
        return hawkes_exp_simulate_by_composition_N(self.theta, N)
    
    def compensator(self, t, history):
        return exp_hawkes_compensators(history, self.theta)

    def intensity(self, t, history):
        return hawkes_exp_intensity(t, history, self.theta)