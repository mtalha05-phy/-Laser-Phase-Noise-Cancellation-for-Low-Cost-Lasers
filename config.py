"""
Configuration module for FT-OPS (Fiber-Optic Transmission Optical Phase Sensing)
Phase Noise Mitigation Simulation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    # Time and Sampling Parameters
    fs: float = 100_000.0          # Sampling frequency (Hz) = 100 kSa/s
    duration: float = 0.5          # Total simulation time (s)
    
    # Laser Parameters
    linewidth: float = 500_000.0   # Laser 3-dB linewidth (Hz) - e.g., 500 kHz standard DFB laser
    wavelength: float = 1550e-9    # Laser center wavelength (m) = 1550 nm
    optical_power_mw: float = 10.0 # Optical power (mW)
    
    # Intermediate Frequency for Heterodyne Detection
    f_if: float = 20_000.0         # Heterodyne IF carrier frequency (Hz)
    
    # Vibration / Perturbation Signal Parameters
    # Modeling multi-tone low-frequency acoustic/vibration signals
    vib_frequencies: tuple = (25.0, 75.0, 150.0)    # Frequencies (Hz)
    vib_amplitudes: tuple = (0.5, 0.3, 0.2)        # Peak phase amplitudes (rad)
    vib_phases: tuple = (0.0, np.pi/4, np.pi/2)    # Initial phase offsets (rad)
    
    # Additive Channel / Photodetector Noise
    snr_det_db: float = 30.0       # Photodetector SNR (dB) for thermal/shot/ASE noise
    
    # Delayed Self-Heterodyne (DSH) Parameters
    # Fiber delay line: length L_delay -> tau_delay = n * L / c
    fiber_refractive_index: float = 1.4682
    delay_line_length_m: float = 2000.0  # 2 km fiber delay line
    
    # Wiener Filter Parameters
    wiener_filter_order: int = 128       # FIR filter length for time-domain Wiener
    
    @property
    def num_samples(self) -> int:
        return int(self.fs * self.duration)
        
    @property
    def t(self) -> np.ndarray:
        return np.arange(self.num_samples) / self.fs
        
    @property
    def tau_delay(self) -> float:
        c = 299_792_458.0  # Speed of light in vacuum (m/s)
        return (self.fiber_refractive_index * self.delay_line_length_m) / c

    @property
    def delay_samples(self) -> int:
        # Delay rounded to nearest integer samples
        return max(1, int(round(self.tau_delay * self.fs)))
