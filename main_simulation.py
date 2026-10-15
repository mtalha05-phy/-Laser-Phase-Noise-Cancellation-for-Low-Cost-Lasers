"""
Main Simulation Script for FT-OPS Laser Phase Noise Mitigation.

Compares:
1. Raw Unfiltered Demodulated Phase (Baseline)
2. Optimal Wiener Filtering (Frequency & Time Domain)
3. Single-Channel Delayed Self-Heterodyne (DSH) Deconvolution
4. Reference-Assisted Dual-Arm DSH Cancellation
"""

import sys
import os
import numpy as np

from config import SimulationConfig
from laser_model import generate_laser_phase_noise, generate_vibration_signal
from receiver_model import DigitalReceiver
from wiener_filter import WienerPhaseFilter
from dsh_processor import DelayedSelfHeterodyneProcessor
from evaluation import compute_metrics, plot_comprehensive_comparison


def run_full_simulation(seed: int = 42):
    print("=" * 70)
    print("   FT-OPS Laser Phase Noise Mitigation Simulation Benchmark   ")
    print("=" * 70)
    
    # 1. Initialize Simulation Parameters
    cfg = SimulationConfig(
        fs=100_000.0,            # 100 kSa/s sampling rate
        duration=0.5,            # 0.5 seconds observation
        linewidth=500_000.0,     # 500 kHz commercial DFB laser linewidth
        f_if=20_000.0,           # 20 kHz heterodyne carrier
        snr_det_db=32.0,         # 32 dB photodetector SNR
        delay_line_length_m=2000.0 # 2 km fiber delay line for DSH
    )
    
    print(f"[*] Configuration:")
    print(f"    - Laser Linewidth:     {cfg.linewidth / 1e3:.1f} kHz (Broad-linewidth Commercial Laser)")
    print(f"    - Sampling Frequency:  {cfg.fs / 1e3:.1f} kSa/s")
    print(f"    - Fiber Delay Line:    {cfg.delay_line_length_m:.1f} m (tau_d = {cfg.tau_delay * 1e6:.2f} us)")
    print(f"    - Vibration Tones:     {cfg.vib_frequencies} Hz")
    print(f"    - Photodetector SNR:   {cfg.snr_det_db:.1f} dB")
    print("-" * 70)
    
    # 2. Generate Signals
    print("[*] Step 1: Synthesizing Laser Phase Noise and Vibration Perturbation...")
    phi_laser = generate_laser_phase_noise(cfg, seed=seed)
    phi_vib_true = generate_vibration_signal(cfg)
    
    # Sensing arm total phase
    phi_total_sens = phi_laser + phi_vib_true
    
    # Reference arm phase (monitors pure laser phase via local reference path)
    phi_total_ref = phi_laser
    
    # 3. Simulate Digital Receiver Detection & Phase Demodulation
    print("[*] Step 2: Simulating Optical Heterodyne Detection & I/Q Demodulation...")
    receiver = DigitalReceiver(cfg)
    
    # Heterodyne detection on sensing arm
    i_het_sens, t = receiver.simulate_heterodyne_detection(phi_total_sens, seed=seed)
    phi_raw_sens = receiver.demodulate_phase(i_het_sens)
    
    # Heterodyne detection on reference arm
    i_het_ref, _ = receiver.simulate_heterodyne_detection(phi_total_ref, seed=seed + 100)
    phi_raw_ref = receiver.demodulate_phase(i_het_ref)
    
    # 4. Method 1: Wiener Filtering
    print("[*] Step 3: Applying Wiener Filtering (Phase Noise Spectrum Shaping)...")
    wiener_filter = WienerPhaseFilter(cfg)
    phi_wiener, H_wiener, freqs_w = wiener_filter.frequency_domain_filter(
        phi_raw_sens,
        linewidth=cfg.linewidth,
        vib_band=(10.0, 300.0)
    )
    
    # 5. Method 2: Delayed Self-Heterodyne (DSH) Processing
    print("[*] Step 4: Applying Delayed Self-Heterodyne (DSH) Processing...")
    dsh_proc = DelayedSelfHeterodyneProcessor(cfg)
    
    # 5a. Single-Channel DSH with Tikhonov Regularized Deconvolution
    delta_phi_single = dsh_proc.compute_differential_phase(phi_raw_sens)
    phi_dsh_single, _, _ = dsh_proc.deconvolve_single_channel_dsh(
        delta_phi_single, tikhonov_alpha=0.03
    )
    
    # 5b. Reference-Assisted Dual-Arm DSH Cancellation
    phi_dsh_ref, _ = dsh_proc.reference_assisted_dsh_cancellation(
        phi_raw_sens, phi_raw_ref, tikhonov_alpha=0.02
    )
    
    # 6. Quantitative Evaluation & Metrics
    print("[*] Step 5: Computing Performance Metrics...")
    metrics_raw = compute_metrics(phi_vib_true, phi_raw_sens)
    metrics_wiener = compute_metrics(phi_vib_true, phi_wiener)
    metrics_dsh_s = compute_metrics(phi_vib_true, phi_dsh_single)
    metrics_dsh_r = compute_metrics(phi_vib_true, phi_dsh_ref)
    
    print("\n" + "=" * 70)
    print(f"{'Method / Processing Pipeline':<30} | {'SNR (dB)':<10} | {'RMSE (rad)':<12} | {'Corr (rho)':<10}")
    print("-" * 70)
    print(f"{'1. Raw Demodulated Phase':<30} | {metrics_raw['snr_db']:<10.2f} | {metrics_raw['rmse']:<12.4f} | {metrics_raw['correlation']:<10.4f}")
    print(f"{'2. Wiener Filter':<30} | {metrics_wiener['snr_db']:<10.2f} | {metrics_wiener['rmse']:<12.4f} | {metrics_wiener['correlation']:<10.4f}")
    print(f"{'3. Single-Channel DSH':<30} | {metrics_dsh_s['snr_db']:<10.2f} | {metrics_dsh_s['rmse']:<12.4f} | {metrics_dsh_s['correlation']:<10.4f}")
    print(f"{'4. Reference-Assisted DSH':<30} | {metrics_dsh_r['snr_db']:<10.2f} | {metrics_dsh_r['rmse']:<12.4f} | {metrics_dsh_r['correlation']:<10.4f}")
    print("=" * 70)
    
    # 7. Generate Visualizations
    out_img = os.path.join(os.path.dirname(__file__), "ft_ops_comparison.png")
    plot_comprehensive_comparison(
        t=t,
        phi_vib_true=phi_vib_true,
        phi_raw=phi_raw_sens,
        phi_wiener=phi_wiener,
        phi_dsh_single=phi_dsh_single,
        phi_dsh_ref=phi_dsh_ref,
        fs=cfg.fs,
        save_path=out_img
    )
    print(f"\n[+] Simulation complete. Report figure generated at: {out_img}\n")
    return {
        "raw": metrics_raw,
        "wiener": metrics_wiener,
        "dsh_single": metrics_dsh_s,
        "dsh_ref": metrics_dsh_r
    }


if __name__ == "__main__":
    run_full_simulation()
