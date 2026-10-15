"""
Digital Coherent/Heterodyne Receiver and Phase Demodulation Module.

Simulates:
1. Optical heterodyne mixing of sensing optical carrier with local oscillator.
2. Photodetection with additive thermal/shot/ASE detector noise.
3. Digital downconversion to In-phase (I) and Quadrature (Q) baseband components.
4. Anti-aliasing low-pass filtering.
5. Arctangent phase extraction and phase unwrapping.
"""

import numpy as np
from scipy import signal
from typing import Tuple
from config import SimulationConfig


class DigitalReceiver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Design a Butterworth low-pass filter for I/Q downconversion
        # Cutoff set appropriately below the 2*f_if image frequency
        cutoff_hz = min(config.f_if * 0.8, config.fs * 0.4)
        order = 4
        self.b_lpf, self.a_lpf = signal.butter(
            order, cutoff_hz / (config.fs / 2.0), btype='low'
        )

    def simulate_heterodyne_detection(
        self, phi_total: np.ndarray, seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates optical heterodyne detection at intermediate frequency f_if.
        
        I_het(t) = A * cos(2 * pi * f_if * t + phi_total(t)) + n_det(t)
        
        Returns:
            i_het: Detected RF heterodyne photocurrent signal.
            t: Time vector.
        """
        if seed is not None:
            np.random.seed(seed)
            
        t = self.config.t
        carrier_phase = 2.0 * np.pi * self.config.f_if * t + phi_total
        carrier = np.cos(carrier_phase)
        
        # Additive white Gaussian noise (photodetector thermal/shot noise)
        signal_power = np.mean(carrier ** 2)
        noise_power = signal_power / (10.0 ** (self.config.snr_det_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(t))
        
        i_het = carrier + noise
        return i_het, t

    def demodulate_phase(self, i_het: np.ndarray) -> np.ndarray:
        """
        Performs digital I/Q downconversion, low-pass filtering,
        arctangent phase extraction, and phase unwrapping.
        
        Returns:
            phi_demod: Unwrapped extracted phase in radians.
        """
        t = self.config.t
        
        # In-phase and Quadrature mixing
        i_mixed = 2.0 * i_het * np.cos(2.0 * np.pi * self.config.f_if * t)
        q_mixed = -2.0 * i_het * np.sin(2.0 * np.pi * self.config.f_if * t)
        
        # Zero-phase digital low-pass filtering to remove 2*f_if harmonics
        i_base = signal.filtfilt(self.b_lpf, self.a_lpf, i_mixed)
        q_base = signal.filtfilt(self.b_lpf, self.a_lpf, q_mixed)
        
        # Complex baseband analytic signal
        complex_baseband = i_base + 1j * q_base
        
        # Phase extraction and unwrapping
        raw_phase = np.angle(complex_baseband)
        phi_demod = np.unwrap(raw_phase)
        
        # Remove arbitrary initial DC phase offset
        phi_demod -= np.mean(phi_demod[:100])
        return phi_demod
