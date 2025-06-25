clc;
clear;
close all;

%% Parameters
num_frames = 100; % Number of frames
frame_size = 1000; % Frame size (number of symbols per frame)
num_symbols_for_estimation = 400; % Number of symbols used for parameter estimation
static_channel = false;
M = 13; % Memory length for MP
P = 7; % Maximum nonlinear order for MP

%% Generate data
% [tx_frames, rx_frames] = generate_frames(num_frames, frame_size, static_channel);
load("datafile.mat");
% Initialization
frame_sic = zeros(1, num_frames); % SIC improvement for each frame

%% Apply Memory Polynomial for each frame
for frame_idx = 1:num_frames
    % Extract transmitted and received data for current frame
    txdata = txdata_frames(:, frame_idx);
    rxdata = rxdata_frames(:, frame_idx);

    % Use only the first num_symbols_for_estimation symbols for parameter estimation
    txdata_est = txdata(1:num_symbols_for_estimation);
    rxdata_est = rxdata(1:num_symbols_for_estimation);

    % Construct feature matrix for parameter estimation
    n_features = M * P * (P + 1) / 2;
    X_est = zeros(num_symbols_for_estimation, n_features);

    feature_idx = 1;
    for delay = 0:M-1
        for order = 1:2:P
            for q = 0:order
                for i = 1:num_symbols_for_estimation
                    idx = i - delay;
                    if idx > 0 && idx <= num_symbols_for_estimation
                        X_est(i, feature_idx) = (txdata_est(idx)^(order-q)) * (conj(txdata_est(idx))^q);
                    end
                end
                feature_idx = feature_idx + 1;
            end
        end
    end

    % Compute MP coefficients using only the first num_symbols_for_estimation symbols
    lambda = 1e-6; % Regularization parameter
    c = (X_est' * X_est + lambda * eye(size(X_est, 2))) \ (X_est' * rxdata_est);

    % Construct feature matrix for the full frame
    X_full = zeros(frame_size, n_features);
    feature_idx = 1;
    for delay = 0:M-1
        for order = 1:2:P
            for q = 0:order
                for i = 1:frame_size
                    idx = i - delay;
                    if idx > 0 && idx <= frame_size
                        X_full(i, feature_idx) = (txdata(idx)^(order-q)) * (conj(txdata(idx))^q);
                    end
                end
                feature_idx = feature_idx + 1;
            end
        end
    end

    % Estimate self-interference and perform SIC for the full frame
    si_est = X_full * c; % Self-interference estimate
    rx_clean = rxdata - si_est; % After SIC

    % Compute SIC improvement (in dB) for the full frame
    power_before = mean(abs(rxdata).^2); % Power before SIC
    power_after = mean(abs(rx_clean).^2); % Power after SIC
    frame_sic(frame_idx) = 10 * log10(power_before / power_after); % SIC improvement
end
frame_sic_plot = frame_sic(11:end);
%% Plot SIC improvement over frames
figure;
plot(11:num_frames, frame_sic_plot, '-o', 'LineWidth', 1.5);
xlabel('Frame Number');
ylabel('SIC Improvement (dB)');
title('SIC Improvement vs. Frame Number');
xlim([1, num_frames]);
ylim([-20, 30]);
grid on;
save('ploy90_400','frame_sic_plot');

%% Compute mean and variance of SIC improvement
sic_mean = mean(frame_sic); % Mean of SIC improvement
sic_variance = var(frame_sic); % Variance of SIC improvement

% Display results
fprintf('Mean of SIC Improvement: %.2f dB\n', sic_mean);
fprintf('Variance of SIC Improvement: %.4f dB^2\n', sic_variance);





