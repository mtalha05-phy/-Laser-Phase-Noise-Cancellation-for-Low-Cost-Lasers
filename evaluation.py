"""
Performance Evaluation, Metrics, and Visualization Module for FT-OPS.

Provides:
- Metrics: SNR (dB), RMSE (rad), Correlation Coefficient (rho).
- Power Spectral Density (PSD) estimation using Welch's method.
- Multi-panel publication-quality Matplotlib figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Tuple


def compute_metrics(s_true: np.ndarray, s_est: np.ndarray) -> Dict[str, float]:
    """
    Computes DSP performance metrics between true and estimated signals.
    
    Returns dictionary with:
    - rmse: Root Mean Square Error (rad)
    - snr_db: Signal-to-Noise Ratio (dB)
    - correlation: Pearson correlation coefficient
    """
    # Align lengths if needed
    n = min(len(s_true), len(s_est))
    y_true = s_true[:n]
    y_est = s_est[:n]
    
    # Remove mean offsets
    y_true = y_true - np.mean(y_true)
    y_est = y_est - np.mean(y_est)
    
    # Optimal scale factor alpha to account for linear filtering attenuation
    alpha = np.sum(y_true * y_est) / (np.sum(y_est ** 2) + 1e-12)
    y_est_scaled = alpha * y_est
    
    # Error and RMSE
    err = y_true - y_est_scaled
    rmse = np.sqrt(np.mean(err ** 2))
    
    # SNR in dB
    sig_power = np.mean(y_true ** 2)
    noise_power = np.mean(err ** 2) + 1e-15
    snr_db = 10.0 * np.log10(sig_power / noise_power)
    
    # Pearson Correlation Coefficient
    norm_true = np.linalg.norm(y_true)
    norm_est = np.linalg.norm(y_est)
    if norm_true > 0 and norm_est > 0:
        corr = float(np.dot(y_true, y_est) / (norm_true * norm_est))
    else:
        corr = 0.0
        
    return {
        "rmse": float(rmse),
        "snr_db": float(snr_db),
        "correlation": float(corr),
        "scale_factor": float(alpha)
    }


def compute_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Welch Power Spectral Density in dB (rad^2 / Hz)."""
    nperseg_actual = min(len(x), nperseg)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg_actual, scaling='density')
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd_db


def plot_comprehensive_comparison(
    t: np.ndarray,
    phi_vib_true: np.ndarray,
    phi_raw: np.ndarray,
    phi_wiener: np.ndarray,
    phi_dsh_single: np.ndarray,
    phi_dsh_ref: np.ndarray,
    fs: float,
    save_path: str = "ft_ops_comparison.png"
):
    """
    Generates a 4-panel comprehensive comparative plot:
    1. Raw Phase vs True Vibration (illustrates laser phase drift masking signal)
    2. Recovered Waveforms (Wiener vs DSH vs True Vibration)
    3. Power Spectral Density (PSD) comparison across frequencies
    4. Quantitative Performance Metrics comparison
    """
    metrics_raw = compute_metrics(phi_vib_true, phi_raw)
    metrics_wiener = compute_metrics(phi_vib_true, phi_wiener)
    metrics_dsh_single = compute_metrics(phi_vib_true, phi_dsh_single)
    metrics_dsh_ref = compute_metrics(phi_vib_true, phi_dsh_ref)
    
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    # --- Subplot 1: Raw Phase vs Ground Truth Vibration ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t * 1000, phi_raw, color='gray', alpha=0.7, label='Raw Demodulated Phase (Laser Drift Dominates)')
    ax1.plot(t * 1000, phi_vib_true, color='black', linewidth=2.0, label='True Vibration Signal $\\phi_{vib}(t)$')
    ax1.set_title("Time Domain: Optical Demodulated Phase with Broad-Linewidth Laser Noise", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Phase (rad)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # --- Subplot 2: Filtered / Recovered Waveforms (Zoomed View) ---
    ax2 = fig.add_subplot(gs[1, :])
    # Display 50 ms window for clear waveform inspection
    zoom_samples = min(len(t), int(0.06 * fs))
    t_zoom = t[:zoom_samples] * 1000
    
    ax2.plot(t_zoom, phi_vib_true[:zoom_samples], 'k-', linewidth=2.5, label='Ground Truth Vibration')
    ax2.plot(t_zoom, metrics_wiener['scale_factor'] * phi_wiener[:zoom_samples], 'b--', linewidth=1.8, label=f"Wiener Filter (SNR: {metrics_wiener['snr_db']:.1f} dB)")
    ax2.plot(t_zoom, metrics_dsh_single['scale_factor'] * phi_dsh_single[:zoom_samples], 'r-.', linewidth=1.5, label=f"Single-Channel DSH (SNR: {metrics_dsh_single['snr_db']:.1f} dB)")
    ax2.plot(t_zoom, metrics_dsh_ref['scale_factor'] * phi_dsh_ref[:zoom_samples], 'g-', linewidth=1.8, label=f"Ref-Assisted DSH (SNR: {metrics_dsh_ref['snr_db']:.1f} dB)")
    
    ax2.set_title("Recovered Vibration Waveforms (Zoomed 60 ms View)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude (rad)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')
    
    # --- Subplot 3: Frequency Domain PSD Comparison ---
    ax3 = fig.add_subplot(gs[2, 0])
    f_raw, psd_raw = compute_psd(phi_raw, fs)
    f_true, psd_true = compute_psd(phi_vib_true, fs)
    f_w, psd_w = compute_psd(phi_wiener, fs)
    f_dsh, psd_dsh = compute_psd(phi_dsh_ref, fs)
    
    ax3.semilogx(f_raw, psd_raw, color='gray', alpha=0.6, label='Raw Phase ($1/f^2$ noise floor)')
    ax3.semilogx(f_true, psd_true, 'k-', linewidth=2.0, label='Ground Truth Vibration')
    ax3.semilogx(f_w, psd_w, 'b--', linewidth=1.5, label='Wiener Filtered')
    ax3.semilogx(f_dsh, psd_dsh, 'g-', linewidth=1.5, label='Ref-Assisted DSH')
    
    ax3.set_xlim(5, 1000)
    ax3.set_title("Power Spectral Density (PSD)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (dB rad$^2$/Hz)")
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=9)
    
    # --- Subplot 4: Quantitative Performance Metrics ---
    ax4 = fig.add_subplot(gs[2, 1])
    methods = ['Raw Phase', 'Wiener Filter', 'Single DSH', 'Ref DSH']
    snrs = [metrics_raw['snr_db'], metrics_wiener['snr_db'], metrics_dsh_single['snr_db'], metrics_dsh_ref['snr_db']]
    corrs = [metrics_raw['correlation'], metrics_wiener['correlation'], metrics_dsh_single['correlation'], metrics_dsh_ref['correlation']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    color_snr = '#1f77b4'
    color_corr = '#2ca02c'
    
    bars1 = ax4.bar(x - width/2, snrs, width, label='Output SNR (dB)', color=color_snr)
    ax4.set_ylabel('SNR (dB)', color=color_snr, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor=color_snr)
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=15, ha='right')
    
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, corrs, width, label='Correlation ($\\rho$)', color=color_corr)
    ax4_twin.set_ylabel('Correlation Coefficient $\\rho$', color=color_corr, fontweight='bold')
    ax4_twin.tick_params(axis='y', labelcolor=color_corr)
    ax4_twin.set_ylim(0.0, 1.05)
    
    # Annotate bars
    for b in bars1:
        h = b.get_height()
        ax4.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color=color_snr, fontweight='bold')
    for b in bars2:
        h = b.get_height()
        ax4_twin.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=9, color=color_corr, fontweight='bold')
                          
    ax4.set_title("Performance Metrics Comparison", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved comprehensive plot to: {save_path}")
