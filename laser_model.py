"""
Laser Phase Noise and Vibration Signal Generation Module.

Physics Model:
In semiconductor and solid-state lasers, spontaneous emission into the lasing
mode adds random field fluctuations, leading to white frequency noise.
The laser optical phase undergoes a Wiener process (Brownian random walk):
    d phi_L(t) = sqrt(2 * pi * Delta_nu) * dW(t)
where:
    Delta_nu is the Full Width at Half Maximum (FWHM) 3-dB Lorentzian linewidth,
    dW(t) is standard Brownian motion.

Over discrete sampling intervals T_s = 1 / f_s:
    Delta phi_L[n] ~ N(0, 2 * pi * Delta_nu * T_s)
    phi_L[n] = sum_{k=0}^n Delta phi_L[k]

The corresponding Phase Noise Power Spectral Density (PSD) exhibits a 1/f^2 slope:
    S_{phi}(f) = Delta_nu / (pi * f^2)
"""

import numpy as np
from typing import Tuple
from config import SimulationConfig


def generate_laser_phase_noise(config: SimulationConfig, seed: int = None) -> np.ndarray:
    """
    Generates laser phase noise phi_L(t) as a discrete-time Wiener process.
    
    Args:
        config: SimulationConfig object containing linewidth, fs, and duration.
        seed: Optional random seed for reproducible simulations.
        
    Returns:
        phi_laser: 1D numpy array of instantaneous phase noise in radians.
    """
    if seed is not None:
        np.random.seed(seed)
        
    ts = 1.0 / config.fs
    # Variance of the phase increment per sample
    var_delta_phi = 2.0 * np.pi * config.linewidth * ts
    std_delta_phi = np.sqrt(var_delta_phi)
    
    # White Gaussian phase increments
    delta_phi = np.random.normal(0.0, std_delta_phi, config.num_samples)
    
    # Cumulative integration yields the Wiener process (Brownian random walk)
    phi_laser = np.cumsum(delta_phi)
    
    # Zero-center the initial phase
    phi_laser -= phi_laser[0]
    return phi_laser


def generate_vibration_signal(config: SimulationConfig) -> np.ndarray:
    """
    Generates a realistic multi-tone low-frequency acoustic/vibration phase signal.
    
    phi_vib(t) = sum_m A_m * sin(2 * pi * f_m * t + theta_m)
    
    Args:
        config: SimulationConfig object containing vibration parameters.
        
    Returns:
        phi_vib: 1D numpy array of true vibration phase perturbation in radians.
    """
    t = config.t
    phi_vib = np.zeros_like(t)
    
    for f, a, p in zip(config.vib_frequencies, config.vib_amplitudes, config.vib_phases):
        phi_vib += a * np.sin(2.0 * np.pi * f * t + p)
        
    return phi_vib


def theoretical_phase_noise_psd(f: np.ndarray, linewidth: float) -> np.ndarray:
    """
    Calculates the single-sided theoretical phase noise PSD (rad^2 / Hz).
    S_{phi}(f) = 2 * Delta_nu / (2 * pi * f^2) = Delta_nu / (pi * f^2)
    """
    f_safe = np.where(f == 0, np.nan, f)
    psd = linewidth / (np.pi * (f_safe ** 2))
    return psd
