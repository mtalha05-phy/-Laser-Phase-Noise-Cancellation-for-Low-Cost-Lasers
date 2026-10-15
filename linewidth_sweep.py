"""
Linewidth Sensitivity Sweep Study for FT-OPS.

Investigates:
How does performance scale as laser linewidth transitions from:
- Ultra-Narrow Linewidth Lasers (NLL): 1 kHz - 10 kHz (High Cost)
to
- Standard Commercial DFB Lasers: 100 kHz - 5 MHz (Low Cost)

Plots Output SNR (dB) vs. Laser Linewidth for:
1. Raw Phase Demodulation
2. Wiener Filter
3. Single-Channel DSH
4. Reference-Assisted DSH
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import SimulationConfig
from laser_model import generate_laser_phase_noise, generate_vibration_signal
from receiver_model import DigitalReceiver
from wiener_filter import WienerPhaseFilter
from dsh_processor import DelayedSelfHeterodyneProcessor
from evaluation import compute_metrics


def run_linewidth_sweep():
    print("=" * 70)
    print("   FT-OPS Linewidth Sensitivity Sweep Study   ")
    print("=" * 70)
    
    # Linewidth range: 1 kHz to 5 MHz (log spaced)
    linewidths = np.array([1e3, 5e3, 2e4, 1e5, 5e5, 1e6, 2e6, 5e6])
    
    snr_raw_list = []
    snr_wiener_list = []
    snr_dsh_single_list = []
    snr_dsh_ref_list = []
    
    cfg = SimulationConfig(
        fs=100_000.0,
        duration=0.3,
        f_if=20_000.0,
        snr_det_db=32.0,
        delay_line_length_m=2000.0
    )
    
    receiver = DigitalReceiver(cfg)
    wiener = WienerPhaseFilter(cfg)
    dsh = DelayedSelfHeterodyneProcessor(cfg)
    
    phi_vib_true = generate_vibration_signal(cfg)
    
    for lw in linewidths:
        print(f"[*] Simulating Linewidth = {lw / 1e3:.1f} kHz...")
        cfg.linewidth = lw
        
        # Monte-Carlo iterations for robust averaging
        num_trials = 3
        temp_raw, temp_w, temp_dsh_s, temp_dsh_r = [], [], [], []
        
        for trial in range(num_trials):
            seed = int(lw % 10000) + trial * 37
            phi_laser = generate_laser_phase_noise(cfg, seed=seed)
            phi_sens = phi_laser + phi_vib_true
            phi_ref = phi_laser
            
            # Heterodyne demodulation
            i_het_sens, _ = receiver.simulate_heterodyne_detection(phi_sens, seed=seed)
            phi_raw_sens = receiver.demodulate_phase(i_het_sens)
            
            i_het_ref, _ = receiver.simulate_heterodyne_detection(phi_ref, seed=seed + 50)
            phi_raw_ref = receiver.demodulate_phase(i_het_ref)
            
            # Wiener
            phi_w, _, _ = wiener.frequency_domain_filter(phi_raw_sens, linewidth=lw)
            
            # DSH Single
            delta_s = dsh.compute_differential_phase(phi_raw_sens)
            phi_ds, _, _ = dsh.deconvolve_single_channel_dsh(delta_s, tikhonov_alpha=0.03)
            
            # DSH Ref
            phi_dr, _ = dsh.reference_assisted_dsh_cancellation(phi_raw_sens, phi_raw_ref)
            
            temp_raw.append(compute_metrics(phi_vib_true, phi_raw_sens)['snr_db'])
            temp_w.append(compute_metrics(phi_vib_true, phi_w)['snr_db'])
            temp_dsh_s.append(compute_metrics(phi_vib_true, phi_ds)['snr_db'])
            temp_dsh_r.append(compute_metrics(phi_vib_true, phi_dr)['snr_db'])
            
        snr_raw_list.append(np.mean(temp_raw))
        snr_wiener_list.append(np.mean(temp_w))
        snr_dsh_single_list.append(np.mean(temp_dsh_s))
        snr_dsh_ref_list.append(np.mean(temp_dsh_r))
        
    # Plotting Linewidth Sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lw_khz = linewidths / 1e3
    ax.semilogx(lw_khz, snr_raw_list, 'o--', color='gray', linewidth=2.0, label='Raw Demodulated Phase')
    ax.semilogx(lw_khz, snr_wiener_list, 's-', color='#1f77b4', linewidth=2.5, label='Wiener Filter')
    ax.semilogx(lw_khz, snr_dsh_single_list, '^-.', color='#d62728', linewidth=2.0, label='Single-Channel DSH')
    ax.semilogx(lw_khz, snr_dsh_ref_list, 'd-', color='#2ca02c', linewidth=2.5, label='Reference-Assisted DSH')
    
    # Highlight commercial vs NLL laser domains
    ax.axvspan(1, 10, color='blue', alpha=0.08, label='Ultra-Narrow Linewidth Laser (NLL)')
    ax.axvspan(100, 5000, color='orange', alpha=0.08, label='Commercial DFB Laser Domain')
    
    ax.set_title("Vibration Recovery Performance vs. Laser Linewidth", fontsize=14, fontweight='bold')
    ax.set_xlabel("Laser Linewidth $\\Delta\\nu$ (kHz)", fontsize=12)
    ax.set_ylabel("Output SNR (dB)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10)
    
    out_path = os.path.join(os.path.dirname(__file__), "ft_ops_linewidth_sweep.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] Linewidth sweep complete. Figure saved to: {out_path}")


if __name__ == "__main__":
    run_linewidth_sweep()
