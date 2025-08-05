# from hawkesbook import exp_simulate_by_composition
from numba import njit, prange
import numpy.random as rnd
import numpy as np
from tqdm import tqdm
import random
import torch
from synth_tpp.models import HawkesExpTPP, LognormalRenewalTPP, SelfCorrectingTPP

def numba_seed(seed):
    rnd.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_dataset_N(model, Nmin, Nmax, num_sequences=100, key=123):

    numba_seed(key)
    # rnd.seed(key)
    event_list = []
    N_list = []
    for i in tqdm(range(num_sequences), desc="Simulating event lists"):
        N = rnd.randint(Nmin, Nmax)
        ℋ = model.generate_N(N)
        event_list.append(ℋ)
        N_list.append(N)
    return event_list, N_list


def generate_dataset_T(model, Tmin, Tmax, num_sequences=100, key=123):

    numba_seed(key)
    # rnd.seed(key)
    event_list = []
    T_list = []
    for i in tqdm(range(num_sequences), desc="Simulating event lists"):
        T = rnd.uniform(Tmin, Tmax)
        ℋ = model.generate_T(T)
        event_list.append(ℋ)
        T_list.append(T)
    return event_list, T_list



def generator_factory(process_type):
    if process_type == "hawkes1":
        # Example parameters for hawkes1
        theta = np.array([0.2, 0.8, 1.0])
        return HawkesExpTPP(theta)
    elif process_type == "hawkes2":
        # Example parameters for hawkes2
        theta = np.array([0.3, 0.5, 2.5])
        return HawkesExpTPP(theta)
    elif process_type == "lognormal":
        # Example parameters for lognormal
        theta = np.array([1.0, 2.0])
        return LognormalRenewalTPP(theta)
    elif process_type == "self_correcting":
        # Example parameters for self_correcting
        theta = np.array([1.0, 1.0])
        return SelfCorrectingTPP(theta)
    else:
        raise ValueError(f"Unknown process type: {process_type}")