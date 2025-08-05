from synth_tpp.models import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    spacing = 10000
    n_points = 50
    # Hawkes Process parameters: [baseline intensity, excitation, decay]
    theta = np.array([0.2, 0.8, 1.0])
    model_hawkes1 = HawkesExpTPP(theta)
    # Generate 100 event times for Hawkes process
    sequence_hawkes = model_hawkes1.generate_N(n_points)
    # Time grid for intensity calculation
    ts_hawkes = np.linspace(0, max(sequence_hawkes), spacing)
    # Compute intensity at each time in ts_hawkes
    intensity_hawkes = np.array([model_hawkes1.intensity(t, sequence_hawkes) for t in ts_hawkes])

    # Lognormal Renewal Process parameters: [mean, std]
    theta = np.array([1.0, 2.0])
    model_lognormal = LognormalRenewalTPP(theta)
    # Generate 100 event times for Lognormal Renewal process
    sequence_lognormal = model_lognormal.generate_N(n_points)
    # Time grid for intensity calculation
    ts_lognormal = np.linspace(0, max(sequence_lognormal), spacing)
    # Compute intensity at each time in ts_lognormal
    intensity_lognormal = np.array([model_lognormal.intensity(t, sequence_lognormal) for t in ts_lognormal])

    # Self-Correcting Process parameters: [alpha, beta]
    theta = np.array([1.0, 1.0])
    model_self_correcting = SelfCorrectingTPP(theta)
    # Generate 100 event times for Self-Correcting process
    sequence_self_correcting = model_self_correcting.generate_N(n_points)
    # Time grid for intensity calculation
    ts_self_correcting = np.linspace(0, max(sequence_self_correcting), spacing)
    # Compute intensity at each time in ts_self_correcting
    intensity_self_correcting = np.array([model_self_correcting.intensity(t, sequence_self_correcting) for t in ts_self_correcting])

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    hawkes_events, = plt.plot(sequence_hawkes, np.zeros_like(sequence_hawkes), marker='o', linestyle='none', label='Hawkes Events')
    hawkes_intensity, = plt.plot(ts_hawkes, intensity_hawkes, color='red', label='Hawkes Intensity', alpha=0.5)
    plt.xlabel('Time')
    plt.title('Hawkes Process Events')

    plt.subplot(3, 1, 2)
    lognormal_events, = plt.plot(sequence_lognormal, np.zeros_like(sequence_lognormal), marker='o', linestyle='none', color='orange', label='Lognormal Events')
    lognormal_intensity, = plt.plot(ts_lognormal, intensity_lognormal, color='blue', label='Lognormal Intensity', alpha=0.5)
    plt.xlabel('Time')
    plt.title('Lognormal Renewal Process Events')

    plt.subplot(3, 1, 3)
    self_correcting_events, = plt.plot(sequence_self_correcting, np.zeros_like(sequence_self_correcting), marker='o', linestyle='none', color='green', label='Self-Correcting Events')
    self_correcting_intensity, = plt.plot(ts_self_correcting, intensity_self_correcting, color='purple', label='Self-Correcting Intensity', alpha=0.5)
    plt.xlabel('Time')
    plt.title('Self-Correcting Process Events')

    # Create a single legend at the top
    handles = [hawkes_events, hawkes_intensity,
               lognormal_events, lognormal_intensity,
               self_correcting_events, self_correcting_intensity]
    labels = [h.get_label() for h in handles]
    plt.figlegend(handles, labels, loc='upper center', ncol=3, fontsize='large', frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.show()