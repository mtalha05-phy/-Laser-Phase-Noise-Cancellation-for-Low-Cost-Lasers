"""
Delayed Self-Heterodyne (DSH) Processing Module for FT-OPS.

Theory:
When optical phase is processed through a differential delay tau_d:
    Delta_tau phi(t) = phi(t) - phi(t - tau_d)

Transfer Function:
    H_d(f) = 1 - exp(-j * 2 * pi * f * tau_d)
    |H_d(f)| = 2 * |sin(pi * f * tau_d)|

Effect on Laser Phase Noise:
    S_{Delta phi_L}(f) = |H_d(f)|^2 * S_{phi_L}(f)
                       = 4 * sin^2(pi * f * tau_d) * [Delta_nu / (pi * f^2)]
As f -> 0:
    lim_{f -> 0} S_{Delta phi_L}(f) = 4 * pi * Delta_nu * tau_d^2 (Finite and Stationary!)
This completely suppresses the 1/f^2 divergence at DC that masks vibration signals.

This module implements:
1. Single-Channel DSH Differential Phase Extraction & Regularized Inverse Filtering.
2. Dual-Channel Reference-Assisted DSH Phase Noise Cancellation.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from config import SimulationConfig


class DelayedSelfHeterodyneProcessor:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.delay_samples = config.delay_samples
        self.tau_d = config.tau_delay

    def compute_differential_phase(self, phi: np.ndarray, delay_k: Optional[int] = None) -> np.ndarray:
        """
        Computes discrete differential phase:
            Delta_tau phi[n] = phi[n] - phi[n - delay_k]
        """
        if delay_k is None:
            delay_k = self.delay_samples
            
        delta_phi = np.zeros_like(phi)
        delta_phi[delay_k:] = phi[delay_k:] - phi[:-delay_k]
        # For the initial samples before delay, zero or linear extrapolate
        delta_phi[:delay_k] = delta_phi[delay_k]
        return delta_phi

    def deconvolve_single_channel_dsh(
        self,
        delta_phi: np.ndarray,
        tikhonov_alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstructs the original vibration phase phi_vib(t) from differential phase
        Delta_tau phi(t) using regularized frequency-domain inverse filtering.
        
        H_inv(f) = H_d*(f) / (|H_d(f)|^2 + alpha^2)
        
        Args:
            delta_phi: Differential phase array.
            tikhonov_alpha: Regularization parameter to suppress noise amplification
                            at transfer function nulls (f = k / tau_d) and DC.
                            
        Returns:
            phi_hat: Reconstructed vibration phase in time domain.
            H_inv: Inverse filter transfer function.
            freqs: Frequency axis (Hz).
        """
        N = len(delta_phi)
        fs = self.config.fs
        
        # FFT of differential phase
        Delta_Phi = np.fft.rfft(delta_phi)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        
        # Forward transfer function H_d(f) = 1 - exp(-j * 2 * pi * f * tau_d)
        omega = 2.0 * np.pi * freqs
        H_d = 1.0 - np.exp(-1j * omega * self.tau_d)
        
        # Regularized Tikhonov Inverse Filter
        # H_inv = conj(H_d) / (|H_d|^2 + alpha^2)
        H_mag_sq = np.abs(H_d) ** 2
        H_inv = np.conj(H_d) / (H_mag_sq + (tikhonov_alpha ** 2))
        
        # Suppress DC component explicitly
        H_inv[0] = 0.0
        
        # Apply inverse filter in frequency domain
        Phi_hat = Delta_Phi * H_inv
        
        # Bandpass filter to retain acoustic/vibration frequencies (e.g. 5 Hz - 500 Hz)
        bp_mask = (freqs >= 5.0) & (freqs <= 500.0)
        Phi_hat[~bp_mask] *= 0.01
        
        phi_hat = np.fft.irfft(Phi_hat, n=N)
        phi_hat -= np.mean(phi_hat)
        return phi_hat, H_inv, freqs

    def reference_assisted_dsh_cancellation(
        self,
        phi_sensing: np.ndarray,
        phi_ref_laser: np.ndarray,
        tikhonov_alpha: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates dual-arm reference-assisted delayed self-heterodyne architecture.
        
        In this optical design:
        - Sensing channel contains: phi_L(t) + phi_vib(t)
        - Reference channel monitors: phi_L(t) (pure laser noise via local loop)
        
        Differential operation:
            Delta phi_sens(t) = Delta phi_L(t) + Delta phi_vib(t)
            Delta phi_ref(t)  = Delta phi_L(t)
        Cancellation:
            Delta phi_clean(t) = Delta phi_sens(t) - Delta phi_ref(t) = Delta phi_vib(t)
            
        Followed by regularized deconvolution of Delta phi_vib(t).
        
        Returns:
            phi_hat: Reconstructed pristine vibration signal.
            delta_phi_clean: Residual differential phase after cancellation.
        """
        # Step 1: Compute differential phase on both arms
        delta_sens = self.compute_differential_phase(phi_sensing)
        delta_ref = self.compute_differential_phase(phi_ref_laser)
        
        # Step 2: Optical/Digital cancellation of laser phase drift
        delta_phi_clean = delta_sens - delta_ref
        
        # Step 3: Deconvolve clean differential phase to recover vibration signal
        phi_hat, _, _ = self.deconvolve_single_channel_dsh(
            delta_phi_clean, tikhonov_alpha=tikhonov_alpha
        )
        return phi_hat, delta_phi_clean
