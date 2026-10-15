"""
Wiener Filtering Module for Laser Phase Noise Mitigation.

Theory:
Given the observed noisy phase signal:
    y(t) = s(t) + v(t)
where:
    s(t) = phi_vib(t) is the desired low-frequency acoustic/vibration signal
    v(t) = phi_L(t) + n(t) is the laser phase drift + detector noise

1. Frequency-Domain Wiener Filter:
   Non-causal transfer function:
       H(f) = S_ss(f) / (S_ss(f) + S_vv(f))
   where S_ss(f) is the signal PSD, and S_vv(f) is the noise PSD.
   Laser phase noise PSD follows Lorentzian profile:
       S_{phi_L}(f) = Delta_nu / (pi * f^2)

2. Time-Domain FIR Wiener Filter:
   Wiener-Hopf optimal filter:
       R_yy * h = r_ys
   with symmetric lag delay D = M / 2 to approximate non-causal zero-phase response.
"""

import numpy as np
from scipy import linalg, signal
from typing import Tuple, Optional
from config import SimulationConfig


class WienerPhaseFilter:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def frequency_domain_filter(
        self,
        y: np.ndarray,
        linewidth: Optional[float] = None,
        vib_band: Tuple[float, float] = (10.0, 300.0),
        reg_floor: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies a frequency-domain optimal Wiener filter to the unwrapped phase.
        
        Args:
            y: Observed noisy unwrapped phase signal.
            linewidth: Laser linewidth (Hz). Defaults to config.linewidth.
            vib_band: Frequency range (f_low, f_high) where vibration power is concentrated.
            reg_floor: Regularization floor to avoid division by zero.
            
        Returns:
            s_hat: Estimated vibration signal in time domain.
            H_f: Frequency response of the Wiener filter.
            freqs: Frequency axis (Hz).
        """
        if linewidth is None:
            linewidth = self.config.linewidth
            
        N = len(y)
        fs = self.config.fs
        
        # Remove linear trend / DC ramp from unbounded Brownian drift
        # to ensure spectral consistency for FFT
        detrended_y, slope, intercept = self._detrend_linear(y)
        
        # FFT of observed signal
        Y = np.fft.rfft(detrended_y)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        
        # Model Noise PSD S_vv(f)
        # Laser phase noise: S_laser(f) = Delta_nu / (pi * f^2)
        # Additive photodetector noise floor: sigma_det^2 / fs
        f_safe = np.maximum(freqs, 1.0)  # Avoid DC singularity
        S_laser = linewidth / (np.pi * (f_safe ** 2))
        
        # Estimate detector noise floor from high frequencies (above vibration band)
        high_freq_mask = freqs > (config_cutoff := min(fs * 0.4, 2000.0))
        if np.any(high_freq_mask):
            noise_floor = np.median(np.abs(Y[high_freq_mask]) ** 2) / N
        else:
            noise_floor = 1e-6
            
        S_vv = S_laser + noise_floor
        
        # Observed signal PSD estimate S_yy(f)
        S_yy = (np.abs(Y) ** 2) / N
        
        # Construct Signal PSD S_ss(f) = max(0, S_yy - S_vv) with band prior
        in_band = (freqs >= vib_band[0]) & (freqs <= vib_band[1])
        S_ss = np.maximum(0.0, S_yy - S_vv)
        S_ss[~in_band] *= 0.01  # Suppress out-of-band energy
        
        # Optimal Wiener Transfer Function: H(f) = S_ss / (S_ss + S_vv)
        H_f = S_ss / (S_ss + S_vv + reg_floor)
        H_f = np.clip(H_f, 0.0, 1.0)
        
        # Filter in frequency domain
        S_hat_f = Y * H_f
        s_hat = np.fft.irfft(S_hat_f, n=N)
        
        # Zero-mean the estimated vibration signal
        s_hat -= np.mean(s_hat)
        return s_hat, H_f, freqs

    def time_domain_fir_filter(
        self,
        y: np.ndarray,
        s_ref: np.ndarray,
        filter_order: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves Wiener-Hopf equations R_yy * h = r_ys for optimal FIR filter weights.
        Designed with delay D = M // 2 to yield a symmetric non-causal response.
        
        Args:
            y: Observed noisy signal.
            s_ref: Reference / training vibration signal.
            filter_order: Number of FIR taps (M).
            
        Returns:
            s_hat: Filtered vibration estimate.
            h: Optimal FIR filter impulse response.
        """
        if filter_order is None:
            filter_order = self.config.wiener_filter_order
            
        M = filter_order
        D = M // 2  # Delay for optimal non-causal filtering
        
        # High-pass filter y to remove non-stationary DC drift before autocorrelation
        b_hp, a_hp = signal.butter(2, 5.0 / (self.config.fs / 2.0), btype='high')
        y_hp = signal.filtfilt(b_hp, a_hp, y)
        
        # Compute autocorrelation of y_hp and cross-correlation with s_ref
        N = len(y_hp)
        r_yy_full = signal.correlate(y_hp, y_hp, mode='full') / N
        lags_yy = signal.correlation_lags(N, N)
        zero_lag_idx = np.where(lags_yy == 0)[0][0]
        r_yy = r_yy_full[zero_lag_idx:zero_lag_idx + M]
        
        # Cross-correlation r_ys[k] = E[y[n-k] * s[n-D]]
        r_ys_full = signal.correlate(y_hp, s_ref, mode='full') / N
        lags_ys = signal.correlation_lags(N, len(s_ref))
        r_ys = np.zeros(M)
        for k in range(M):
            lag = -(D - k)
            idx = np.where(lags_ys == lag)[0]
            if len(idx) > 0:
                r_ys[k] = r_ys_full[idx[0]]
                
        # Construct Toeplitz autocorrelation matrix with Tikhonov regularization
        R_yy = linalg.toeplitz(r_yy) + 1e-4 * np.eye(M) * np.trace(linalg.toeplitz(r_yy)) / M
        
        # Solve Wiener-Hopf matrix equation
        h = linalg.solve(R_yy, r_ys)
        
        # Apply filter and compensate for delay D
        y_filt = signal.lfilter(h, 1.0, y_hp)
        s_hat = np.zeros_like(y_filt)
        s_hat[:-D] = y_filt[D:]
        s_hat[-D:] = y_filt[-1]
        
        return s_hat, h

    @staticmethod
    def _detrend_linear(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Removes best-fit linear trend from an array."""
        t = np.arange(len(x))
        p = np.polyfit(t, x, 1)
        trend = np.polyval(p, t)
        return x - trend, p[0], p[1]
