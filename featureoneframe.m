clc;
clear;
close all;

%% Parameters
frame_size = 1000;
num_frames = 1; % One frame for training and evaluation
M_est = 13;
num_symbols_for_estimation = 400;
hidden_units = 17;
nfft = 4096; % FFT length for PSD calculation
fs = 20e6; % Sampling frequency (20 MHz)
lambda = 1e-3;

%% Generate Data for One Frame
[txdata_frames, rxdata_frames] = generate_frames(num_frames, frame_size, true);

%% Linear Cancellation (Least Squares with 400 Symbols)
linear_residual_frames = zeros(size(rxdata_frames));
channel_coefficients = zeros(M_est, num_frames);

txdata_frame = txdata_frames(1:num_symbols_for_estimation, 1);
rxdata_frame = rxdata_frames(1:num_symbols_for_estimation, 1);

num_valid_samples = num_symbols_for_estimation - M_est;
X = zeros(num_valid_samples, M_est);
y = rxdata_frame(M_est+1:num_symbols_for_estimation);

for i = 1:num_valid_samples
    n = i + M_est;
    X(i, :) = txdata_frame(n-M_est+1:n).';
end

h_ls = (X' * X + lambda * eye(size(X' * X))) \ (X' * y);
channel_coefficients(:, 1) = h_ls;

full_X = zeros(frame_size - M_est, M_est);
for i = 1:frame_size - M_est
    n = i + M_est;
    full_X(i, :) = txdata_frames(n-M_est+1:n, 1).';
end

linear_component = zeros(frame_size, 1);
linear_component(M_est+1:frame_size) = full_X * h_ls;
linear_residual_frames(:, 1) = rxdata_frames(:, 1) - linear_component;

%% Training Data Preparation (First Frame)
training_features = [];
training_targets = [];

txdata_train = txdata_frames(:, 1);
linear_residual_train = linear_residual_frames(:, 1);
h_ls = channel_coefficients(:, 1);

tx_real = real(txdata_train);
tx_imag = imag(txdata_train);

for n = M_est:frame_size
    % Historical signal real and imaginary parts
    x_d_real = tx_real(n-M_est+1:n);
    x_d_imag = tx_imag(n-M_est+1:n);

    % Calculating f_i as per the given definition
    f_i = zeros(1, 2 * M_est);
    for i = 1:(2 * M_est)
        idx = n - floor((i - 1) / 2);

        if idx >= 1 && idx <= frame_size
            h_current = h_ls(ceil(i / 2));
            x_d_current = txdata_train(idx);

            if mod(i, 2) == 1  % i is odd -> real part
                f_i(i) = real(h_current * x_d_current);
            else  % i is even -> imaginary part
                f_i(i) = imag(h_current * x_d_current);
            end
        else
            error('Sliding window index out of bounds: idx = %d', idx);
        end
    end

    feature_vector = [x_d_real', x_d_imag', f_i];
    training_features = [training_features; feature_vector];
    training_targets = [training_targets; real(linear_residual_train(n)), imag(linear_residual_train(n))];
end

%% Neural Network Configuration
layers = [
    featureInputLayer(4 * M_est, 'Normalization', 'none', 'Name', 'InputLayer')
    fullyConnectedLayer(hidden_units, 'Name', 'FC1')
    reluLayer('Name', 'ReLU1')
    fullyConnectedLayer(2, 'Name', 'OutputLayer')
    nmseLoss('NMSELoss')  % 使用自定义的 NMSE 损失层
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true);

net = trainNetwork(training_features, training_targets, layers, options);

%% Evaluation (Same Frame)
txdata_eval = txdata_frames(:, 1);
linear_residual_eval = linear_residual_frames(:, 1);

tx_real_eval = real(txdata_eval);
tx_imag_eval = imag(txdata_eval);

input_features_eval = [];

for n = M_est:frame_size
    x_d_real = tx_real_eval(n-M_est+1:n);
    x_d_imag = tx_imag_eval(n-M_est+1:n);

    f_i = zeros(1, 2 * M_est);
    for i = 1:(2 * M_est)
        idx = n - floor((i - 1) / 2);

        if idx >= 1 && idx <= frame_size
            h_current = h_ls(ceil(i / 2));
            x_d_current = txdata_eval(idx);

            if mod(i, 2) == 1
                f_i(i) = real(h_current * x_d_current);
            else
                f_i(i) = imag(h_current * x_d_current);
            end
        else
            error('Sliding window index out of bounds: idx = %d', idx);
        end
    end

    feature_vector = [x_d_real', x_d_imag', f_i];
    input_features_eval = [input_features_eval; feature_vector];
end

predicted_output = predict(net, input_features_eval);
predicted_real = predicted_output(:, 1);
predicted_imag = predicted_output(:, 2);
predicted_si = predicted_real + 1j * predicted_imag;

linear_residual_current_frame = linear_residual_frames(:, 1);
residual_after_SIC = linear_residual_current_frame(M_est:end) - predicted_si;

P_before = sum(abs(linear_residual_frames(:, 1)).^2);
P_after = sum(abs(residual_after_SIC).^2);
C_dB = 10 * log10(P_before / P_after);
disp(['SIC Performance (C in dB): ', num2str(C_dB)]);

%% Compute and Plot PSD
fft_rx = fftshift(fft(rxdata_frames(:, 1), nfft));
fft_after_linear = fftshift(fft(linear_residual_frames(:, 1), nfft));
fft_after_nn = fftshift(fft(residual_after_SIC, nfft));

psd_rx = abs(fft_rx).^2 / nfft;
psd_after_linear = abs(fft_after_linear).^2 / nfft;
psd_after_nn = abs(fft_after_nn).^2 / nfft;

freq = (-nfft/2:nfft/2-1) * fs / nfft;

smooth_psd_rx = smooth(10 * log10(psd_rx), 50);
smooth_psd_after_linear = smooth(10 * log10(psd_after_linear), 50);
smooth_psd_after_nn = smooth(10 * log10(psd_after_nn), 50);

figure;
plot(freq / 1e6, smooth_psd_rx, 'r', 'LineWidth', 1.5); hold on;
plot(freq / 1e6, smooth_psd_after_linear, 'g', 'LineWidth', 1.5);
plot(freq / 1e6, smooth_psd_after_nn, 'b', 'LineWidth', 1.5);
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB)');
title('PSD Before and After SIC');
legend('Received Signal', 'After Linear SIC', 'After NN SIC');
grid on;

%% Compute and Plot PSD for Predicted and Target Signals
fft_target_si = fftshift(fft(linear_residual_current_frame(M_est:end), nfft)); % Target SI after linear cancellation (to be reduced by NN)
fft_predicted_si = fftshift(fft(predicted_si, nfft)); % Predicted SI by NN

% Power Spectral Density calculation
psd_target_si = abs(fft_target_si).^2 / nfft;
psd_predicted_si = abs(fft_predicted_si).^2 / nfft;

% Frequency axis
freq = (-nfft/2:nfft/2-1) * fs / nfft;

% Smooth PSD for better visualization
smooth_psd_target_si = smooth(10 * log10(psd_target_si), 50);
smooth_psd_predicted_si = smooth(10 * log10(psd_predicted_si), 50);

% Plot PSD
figure;
plot(freq / 1e6, smooth_psd_target_si, 'k', 'LineWidth', 1.5); hold on;
plot(freq / 1e6, smooth_psd_predicted_si, 'm', 'LineWidth', 1.5);
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB)');
title('PSD of Target and Predicted Signals');
legend('Target Signal (Linear Residual)', 'Predicted Signal (NN Output)');
grid on;





