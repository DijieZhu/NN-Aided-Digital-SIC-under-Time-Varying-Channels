clc;
clear;
close all;

%% Parameters
num_frames = 10; % Number of frames
frame_size = 1000; % Frame size (number of symbols per frame)
static_channel = false;
M = 13; % Memory length for MP
P = 7; % Maximum nonlinear order for MP

%% Generate data
[tx_frames, rx_frames] = generate_frames(num_frames, frame_size, static_channel);

% Initialization
frame_sic = zeros(1, num_frames); % SIC improvement for each frame

%% Train MP coefficients using the first frame
txdata_train = tx_frames(:, 1); % Transmitted data of first frame
rxdata_train = rx_frames(:, 1); % Received data of first frame

% Construct feature matrix for the first frame
n_samples = length(txdata_train);
n_features = M * P * (P + 1) / 2;
X_train = zeros(n_samples, n_features);

feature_idx = 1;
for delay = 0:M-1
    for order = 1:2:P
        for q = 0:order
            for i = 1:n_samples
                idx = i - delay;
                if idx > 0 && idx <= n_samples
                    X_train(i, feature_idx) = (txdata_train(idx)^(order-q)) * (conj(txdata_train(idx))^q);
                end
            end
            feature_idx = feature_idx + 1;
        end
    end
end

% Compute MP coefficients
lambda = 1e-6; % Regularization parameter
c = (X_train' * X_train + lambda * eye(size(X_train, 2))) \ (X_train' * rxdata_train);

%% Apply the same coefficients to all frames
for frame_idx = 1:num_frames
    txdata = tx_frames(:, frame_idx);
    rxdata = rx_frames(:, frame_idx);

    % Construct feature matrix for the current frame
    X = zeros(n_samples, n_features);

    feature_idx = 1;
    for delay = 0:M-1
        for order = 1:2:P
            for q = 0:order
                for i = 1:n_samples
                    idx = i - delay;
                    if idx > 0 && idx <= n_samples
                        X(i, feature_idx) = (txdata(idx)^(order-q)) * (conj(txdata(idx))^q);
                    end
                end
                feature_idx = feature_idx + 1;
            end
        end
    end

    % Estimate self-interference and perform SIC
    si_est = X * c; % Self-interference estimate
    rx_clean = rxdata - si_est; % After SIC

    % Compute SIC improvement (in dB)
    power_before = mean(abs(rxdata).^2); % Power before SIC
    power_after = mean(abs(rx_clean).^2); % Power after SIC
    frame_sic(frame_idx) = 10 * log10(power_before / power_after); % SIC improvement
end

%% Plot SIC improvement over frames
figure;
plot(1:num_frames, frame_sic, '-o', 'LineWidth', 1.5);
xlabel('Frame Number');
ylabel('SIC Improvement (dB)');
title('SIC Improvement vs. Frame Number');
grid on;

%% Compute mean and variance of SIC improvement
sic_mean = mean(frame_sic); % Mean of SIC improvement
sic_variance = var(frame_sic); % Variance of SIC improvement

% Display results
fprintf('Mean of SIC Improvement: %.2f dB\n', sic_mean);
fprintf('Variance of SIC Improvement: %.4f dB^2\n', sic_variance);















