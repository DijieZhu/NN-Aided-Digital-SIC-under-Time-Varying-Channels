clc;
clear;
close all;

%% Parameters of Memory Ploynomials
load("simulation20Mhz.mat");
M = 13; % Memory length
P = 7;  % Maximum nonlinear order

% Use the entire dataset for SIC
DigitalSI = analogResidual; % Using all analog residual points
txdata = txdata_shaped;  % Using all transmitted samples
ns = length(DigitalSI);    % Number of data points

% Construct the feature matrix
n_features = M * P * (P + 1) / 2; 
X = zeros(ns, n_features);

feature_idx = 1;
for delay = 0:M-1
    for order = 1:2:P
        for q = 0:order
            for i = 1:ns
                idx = i - delay;
                if idx > 0 && idx <= ns
                    X(i, feature_idx) = (txdata(idx)^(order-q)) * (conj(txdata(idx))^q);
                end
            end
            feature_idx = feature_idx + 1;
        end
    end
end

% Regularization and weight computation
lambda = 1e-6; % Regularization strength
h = (X' * X + lambda * eye(size(X, 2))) \ (X' * DigitalSI);

% Reconstruct self-interference signal
y_hat_SI = X * h;

% Self-interference cancellation
y_afterDSIC = DigitalSI - y_hat_SI;

% Compute power spectral density
nfft = 4096;
fs = 20e6;
DigitalSI = DigitalSI - mean(DigitalSI);
y_afterDSIC = y_afterDSIC - mean(y_afterDSIC);
window = hamming(length(DigitalSI));
DigitalSI_windowed = DigitalSI .* window;
y_afterDSIC_windowed = y_afterDSIC .* window;


% Compute power spectral density with smoothing
fft_afterDSIC = fftshift(fft(y_afterDSIC_windowed, nfft));
fft_beforeDSIC = fftshift(fft(DigitalSI_windowed, nfft));

psd_aftersi = abs(fft_afterDSIC).^2 / (length(y_afterDSIC) * fs);
psd_beforesi = abs(fft_beforeDSIC).^2 / (length(DigitalSI) * fs);

% Frequency axis
f = linspace(-fs/2, fs/2, nfft); % Two-sided frequency axis

% Apply smoothing filter
smooth_psd_aftersi = smooth(10*log10(psd_aftersi), 50);
smooth_psd_beforesi = smooth(10*log10(psd_beforesi), 50);

% Plot power spectrum with enhanced aesthetics
figure;
plot(f/1e6, smooth_psd_aftersi, 'b-', 'LineWidth', 2); hold on;
plot(f/1e6, smooth_psd_beforesi, 'r--', 'LineWidth', 2);
xlabel('Frequency (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Power Spectral Density (dB/Hz)', 'FontSize', 12, 'FontWeight', 'bold');
title('Power Spectrum Before and After SI Cancellation', 'FontSize', 14, 'FontWeight', 'bold');
legend('After SI Cancellation', 'Before SI Cancellation', 'Location', 'SouthWest');
xlim([-10 10]); % Focus on ±10 MHz
grid on;


% Compute power of the residual signal before and after SIC
power_before_SIC = sum(abs(DigitalSI).^2);
power_after_SIC = sum(abs(y_afterDSIC).^2);

% Calculate SIC performance (C in dB)
C_dB = 10 * log10(power_before_SIC / power_after_SIC);

% Display the result
disp(['SIC Performance (C): ', num2str(C_dB), ' dB']);











