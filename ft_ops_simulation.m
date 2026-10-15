%% FT-OPS Phase Noise Mitigation Simulation (MATLAB Implementation)
% Problem: Long-haul FT-OPS requires expensive Ultra-Narrow Linewidth Lasers (NLL).
%          Standard commercial lasers introduce phase drift that masks vibration signals.
% Solution: Compare Wiener Filtering vs. Delayed Self-Heterodyne (DSH) Processing.
%
% Clear workspace
clear; close all; clc;

fprintf('=======================================================================\n');
fprintf('     FT-OPS Laser Phase Noise Mitigation Simulation Benchmark          \n');
fprintf('=======================================================================\n');

%% 1. Simulation Parameters
fs = 100e3;               % Sampling rate: 100 kSa/s
T_total = 0.5;            % Total duration: 0.5 s
N = round(fs * T_total);  % Total samples
t = (0:N-1)' / fs;        % Time vector

% Laser Parameters
linewidth = 500e3;        % Laser FWHM Linewidth: 500 kHz (Commercial DFB Laser)
f_if = 20e3;              % Heterodyne Intermediate Frequency: 20 kHz
snr_det_db = 32;          % Photodetector SNR: 32 dB

% Fiber Delay Line Parameters for DSH
fiber_n = 1.4682;         % Refractive index
c = 299792458;            % Speed of light (m/s)
delay_length_m = 2000;    % 2 km fiber delay line
tau_d = (fiber_n * delay_length_m) / c; % Time delay (~9.79 us)
delay_samples = max(1, round(tau_d * fs));

% Target Low-Frequency Vibration Signal Parameters
f_vib = [25, 75, 150];    % Vibration frequencies (Hz)
A_vib = [0.5, 0.3, 0.2];  % Phase amplitudes (rad)
p_vib = [0, pi/4, pi/2];  % Initial phases (rad)

fprintf('[*] Configuration:\n');
fprintf('    - Laser Linewidth:     %.1f kHz (Commercial DFB Laser)\n', linewidth/1e3);
fprintf('    - Sampling Frequency:  %.1f kSa/s\n', fs/1e3);
fprintf('    - Fiber Delay Line:    %.1f m (tau_d = %.2f us)\n', delay_length_m, tau_d*1e6);
fprintf('    - Target Frequencies:  [%d, %d, %d] Hz\n', f_vib(1), f_vib(2), f_vib(3));
fprintf('-----------------------------------------------------------------------\n');

%% 2. Signal Generation (Wiener Process Laser Noise & Vibration)
rng(42); % Seed for reproducibility

% Discrete-time Wiener process phase drift: Delta_phi ~ N(0, 2*pi*linewidth*Ts)
var_delta_phi = 2 * pi * linewidth * (1/fs);
delta_phi = sqrt(var_delta_phi) * randn(N, 1);
phi_laser = cumsum(delta_phi);
phi_laser = phi_laser - phi_laser(1);

% Ground-truth multi-tone vibration signal
phi_vib_true = zeros(N, 1);
for k = 1:length(f_vib)
    phi_vib_true = phi_vib_true + A_vib(k) * sin(2*pi*f_vib(k)*t + p_vib(k));
end

% Sensing optical phase and Reference optical phase
phi_sens_total = phi_laser + phi_vib_true;
phi_ref_total  = phi_laser;

%% 3. Optical Heterodyne Receiver & Digital Phase Demodulation
% RF Photocurrent at IF carrier with detector noise
carrier_sens = cos(2*pi*f_if*t + phi_sens_total);
noise_power = var(carrier_sens) / (10^(snr_det_db/10));
i_het_sens = carrier_sens + sqrt(noise_power) * randn(N, 1);

carrier_ref = cos(2*pi*f_if*t + phi_ref_total);
i_het_ref = carrier_ref + sqrt(noise_power) * randn(N, 1);

% Digital I/Q Downconversion & Demodulation
[b_lpf, a_lpf] = butter(4, min(f_if*0.8, fs*0.4)/(fs/2), 'low');

% Demodulate sensing arm
i_base_sens = filtfilt(b_lpf, a_lpf, 2 * i_het_sens .* cos(2*pi*f_if*t));
q_base_sens = filtfilt(b_lpf, a_lpf, -2 * i_het_sens .* sin(2*pi*f_if*t));
phi_raw_sens = unwrap(atan2(q_base_sens, i_base_sens));
phi_raw_sens = phi_raw_sens - mean(phi_raw_sens(1:min(100, N)));

% Demodulate reference arm
i_base_ref = filtfilt(b_lpf, a_lpf, 2 * i_het_ref .* cos(2*pi*f_if*t));
q_base_ref = filtfilt(b_lpf, a_lpf, -2 * i_het_ref .* sin(2*pi*f_if*t));
phi_raw_ref = unwrap(atan2(q_base_ref, i_base_ref));
phi_raw_ref = phi_raw_ref - mean(phi_raw_ref(1:min(100, N)));

%% 4. Method 1: Frequency-Domain Wiener Filtering
% Remove linear trend from Brownian drift
p_trend = polyfit((1:N)', phi_raw_sens, 1);
phi_detrend = phi_raw_sens - polyval(p_trend, (1:N)');

Y = fft(phi_detrend);
freqs = (0:N-1)' * (fs / N);

% Laser noise PSD model: S_vv(f) = linewidth / (pi * f^2)
f_safe = max(freqs, 1.0);
S_laser = linewidth ./ (pi * (f_safe.^2));
% For two-sided FFT symmetry
S_laser(N/2+2:end) = flipud(S_laser(2:N/2));

S_yy = (abs(Y).^2) / N;
S_vv = S_laser + median(S_yy(freqs > 2000 & freqs < fs/2));

% Construct signal PSD S_ss(f) with band prior (10 Hz - 300 Hz)
S_ss = max(0, S_yy - S_vv);
in_band = (freqs >= 10 & freqs <= 300) | (freqs >= (fs - 300) & freqs <= (fs - 10));
S_ss(~in_band) = S_ss(~in_band) * 0.01;

% Wiener Transfer Function
H_wiener = S_ss ./ (S_ss + S_vv + 1e-4);
H_wiener = max(0, min(1, H_wiener));

phi_wiener = real(ifft(Y .* H_wiener));
phi_wiener = phi_wiener - mean(phi_wiener);

%% 5. Method 2: Delayed Self-Heterodyne (DSH) Processing
% 5a. Single-Channel Differential Phase & Deconvolution
delta_sens = zeros(N, 1);
delta_sens(delay_samples+1:end) = phi_raw_sens(delay_samples+1:end) - phi_raw_sens(1:end-delay_samples);
delta_sens(1:delay_samples) = delta_sens(delay_samples+1);

% Regularized inverse filtering
Delta_Y = fft(delta_sens);
omega = 2 * pi * freqs;
H_d = 1 - exp(-1j * omega * tau_d);
H_inv = conj(H_d) ./ (abs(H_d).^2 + (0.03)^2);
H_inv(1) = 0; % Suppress DC

Phi_dsh_s = Delta_Y .* H_inv;
% Retain vibration band
Phi_dsh_s(~in_band) = Phi_dsh_s(~in_band) * 0.01;
phi_dsh_single = real(ifft(Phi_dsh_s));
phi_dsh_single = phi_dsh_single - mean(phi_dsh_single);

% 5b. Reference-Assisted Dual-Arm DSH Cancellation
delta_ref = zeros(N, 1);
delta_ref(delay_samples+1:end) = phi_raw_ref(delay_samples+1:end) - phi_raw_ref(1:end-delay_samples);
delta_ref(1:delay_samples) = delta_ref(delay_samples+1);

delta_clean = delta_sens - delta_ref; % Noise cancelled in differential domain!
Delta_Clean_Y = fft(delta_clean);
Phi_dsh_r = Delta_Clean_Y .* H_inv;
Phi_dsh_r(~in_band) = Phi_dsh_r(~in_band) * 0.01;
phi_dsh_ref = real(ifft(Phi_dsh_r));
phi_dsh_ref = phi_dsh_ref - mean(phi_dsh_ref);

%% 6. Performance Evaluation Metrics
calc_metrics = @(s_true, s_est) struct(...
    'rmse', sqrt(mean((s_true - (sum(s_true.*s_est)/(sum(s_est.^2)+1e-12))*s_est).^2)), ...
    'snr_db', 10*log10(mean(s_true.^2) / (mean((s_true - (sum(s_true.*s_est)/(sum(s_est.^2)+1e-12))*s_est).^2) + 1e-15)), ...
    'corr', (s_true' * s_est) / (norm(s_true) * norm(s_est) + 1e-12), ...
    'scale', sum(s_true.*s_est)/(sum(s_est.^2)+1e-12));

m_raw    = calc_metrics(phi_vib_true, phi_raw_sens);
m_wiener = calc_metrics(phi_vib_true, phi_wiener);
m_dsh_s  = calc_metrics(phi_vib_true, phi_dsh_single);
m_dsh_r  = calc_metrics(phi_vib_true, phi_dsh_ref);

fprintf('\n=======================================================================\n');
fprintf('%-28s | %-10s | %-12s | %-10s\n', 'Method', 'SNR (dB)', 'RMSE (rad)', 'Corr (rho)');
fprintf('-----------------------------------------------------------------------\n');
fprintf('%-28s | %-10.2f | %-12.4f | %-10.4f\n', '1. Raw Demodulated Phase', m_raw.snr_db, m_raw.rmse, m_raw.corr);
fprintf('%-28s | %-10.2f | %-12.4f | %-10.4f\n', '2. Wiener Filter', m_wiener.snr_db, m_wiener.rmse, m_wiener.corr);
fprintf('%-28s | %-10.2f | %-12.4f | %-10.4f\n', '3. Single-Channel DSH', m_dsh_s.snr_db, m_dsh_s.rmse, m_dsh_s.corr);
fprintf('%-28s | %-10.2f | %-12.4f | %-10.4f\n', '4. Reference-Assisted DSH', m_dsh_r.snr_db, m_dsh_r.rmse, m_dsh_r.corr);
fprintf('=======================================================================\n');

%% 7. Visualization
figure('Position', [100, 100, 1100, 750], 'Color', 'w');

% Subplot 1: Raw Phase vs Ground Truth
subplot(3, 2, [1, 2]);
plot(t*1000, phi_raw_sens, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0); hold on;
plot(t*1000, phi_vib_true, 'k', 'LineWidth', 1.8);
title('Time Domain: Optical Demodulated Phase with Broad-Linewidth Laser Noise', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Phase (rad)');
legend('Raw Demodulated Phase (Laser Drift Dominates)', 'True Vibration \phi_{vib}(t)', 'Location', 'best');
grid on;

% Subplot 2: Recovered Waveforms (Zoomed 60 ms)
subplot(3, 2, [3, 4]);
zoom_N = min(N, round(0.06 * fs));
t_z = t(1:zoom_N) * 1000;
plot(t_z, phi_vib_true(1:zoom_N), 'k', 'LineWidth', 2.2); hold on;
plot(t_z, m_wiener.scale * phi_wiener(1:zoom_N), 'b--', 'LineWidth', 1.5);
plot(t_z, m_dsh_s.scale * phi_dsh_single(1:zoom_N), 'r-.', 'LineWidth', 1.3);
plot(t_z, m_dsh_r.scale * phi_dsh_ref(1:zoom_N), 'g-', 'LineWidth', 1.6);
title('Recovered Vibration Waveforms (Zoomed View)', 'FontWeight', 'bold');
xlabel('Time (ms)'); ylabel('Amplitude (rad)');
legend('Ground Truth', sprintf('Wiener Filter (SNR: %.1f dB)', m_wiener.snr_db), ...
       sprintf('Single DSH (SNR: %.1f dB)', m_dsh_s.snr_db), ...
       sprintf('Ref-Assisted DSH (SNR: %.1f dB)', m_dsh_r.snr_db), 'Location', 'best');
grid on;

% Subplot 3: Power Spectral Density
subplot(3, 2, 5);
[p_raw, f_psd] = pwelch(phi_raw_sens, hamming(2048), 1024, 2048, fs);
[p_true, ~]    = pwelch(phi_vib_true, hamming(2048), 1024, 2048, fs);
[p_w, ~]       = pwelch(phi_wiener, hamming(2048), 1024, 2048, fs);
[p_dsh, ~]     = pwelch(phi_dsh_ref, hamming(2048), 1024, 2048, fs);

semilogx(f_psd, 10*log10(p_raw), 'Color', [0.6 0.6 0.6]); hold on;
semilogx(f_psd, 10*log10(p_true), 'k', 'LineWidth', 1.8);
semilogx(f_psd, 10*log10(p_w), 'b--', 'LineWidth', 1.4);
semilogx(f_psd, 10*log10(p_dsh), 'g-', 'LineWidth', 1.4);
xlim([5, 1000]);
title('Power Spectral Density (PSD)', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('PSD (dB rad^2/Hz)');
legend('Raw Phase (1/f^2 noise)', 'Ground Truth', 'Wiener Filter', 'Ref DSH', 'Location', 'southwest');
grid on;

% Subplot 4: Performance Bar Chart
subplot(3, 2, 6);
methods = {'Raw Phase', 'Wiener', 'Single DSH', 'Ref DSH'};
snrs = [m_raw.snr_db, m_wiener.snr_db, m_dsh_s.snr_db, m_dsh_r.snr_db];
b = bar(snrs, 'FaceColor', [0.12 0.47 0.71]);
set(gca, 'XTickLabel', methods);
title('Output SNR Comparison', 'FontWeight', 'bold');
ylabel('SNR (dB)');
grid on;
for i = 1:length(snrs)
    text(i, snrs(i) + (snrs(i)>=0)*1.5 - (snrs(i)<0)*3, sprintf('%.1f dB', snrs(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

fprintf('\n[+] MATLAB Simulation completed successfully.\n');
