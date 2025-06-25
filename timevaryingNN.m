clc;
clear;
close all;

%% Parameters
frame_size = 1000; % Samples per frame
num_train_frames = 10;  % Frames for training
num_eval_frames = 90;   % Frames for evaluation
M = 26;            % Memory length for polynomial SIC
hidden_units = 17; % Number of hidden units in NN layers
fs = 20e6;         % Sampling frequency (20 MHz)
lambda = 1e-3;     % Regularization parameter for Linear Cancellation

%% Generate Data for 10 Frames (1 Training + 9 Evaluation)
num_frames = num_train_frames + num_eval_frames; % Total frames
[txdata_frames, rxdata_frames] = generate_frames(num_frames, frame_size, false);

%% Perform Analog SIC for All Frames
analog_residual_frames = zeros(size(rxdata_frames));

for frame_idx = 1:num_frames
    txdata_frame = txdata_frames(:, frame_idx);
    rxdata_frame = rxdata_frames(:, frame_idx);

    % Call the perform_analog_sic function
    analog_residual_frames(:, frame_idx) = perform_analog_sic(txdata_frame, rxdata_frame, fs, frame_size);
end

%% Linear Cancellation (Least Squares)
mest = M; % Memory length used in LS estimation
ls_frame_size = 400; % Only use 200 samples for LS estimation
linear_residual_frames = zeros(size(analog_residual_frames));

for frame_idx = 1:num_frames
    txdata_ls_frame = txdata_frames(1:ls_frame_size, frame_idx);
    residual_ls_frame = analog_residual_frames(1:ls_frame_size, frame_idx);

    % Prepare input and target for LS estimation
    X = zeros(ls_frame_size - mest, 2 * mest);
    Y = residual_ls_frame(mest+1:end);

    for n = mest+1:ls_frame_size
        x_real = real(txdata_ls_frame(n-mest:n-1));
        x_imag = imag(txdata_ls_frame(n-mest:n-1));
        X(n-mest, :) = [x_real', x_imag'];
    end

    % Calculate LS coefficients
    h_ls = (X' * X + lambda * eye(size(X, 2))) \ (X' * Y);

    % Predict SI for the current frame
    predicted_si = X * h_ls;

    % Store residual after linear cancellation
    linear_residual_frames(:, frame_idx) = analog_residual_frames(:, frame_idx);
    linear_residual_frames(mest+1:ls_frame_size, frame_idx) = Y - predicted_si;
end

%% Normalization for Training
NormalizationFactor = max(abs(linear_residual_frames(:))); % Compute normalization factor
linear_residual_frames_norm = linear_residual_frames / NormalizationFactor; % Normalize for training

%% Training Data (First Frame)
txdata_train = txdata_frames(:, 1);
linear_residual_train = linear_residual_frames_norm(:, 1);

% Extract real and imaginary parts of training data
tx_real = real(txdata_train);
tx_imag = imag(txdata_train);
rx_real = real(linear_residual_train);
rx_imag = imag(linear_residual_train);

% Prepare inputs and outputs for NN
input_features = [];
output_targets = [];
for n = M:frame_size
    % Input: Real and imaginary parts delayed by M steps
    input_features = [input_features; ...
        tx_real(n-M+1:n)', tx_imag(n-M+1:n)'];
    
    % Output: Real and imaginary parts of residual signal
    output_targets = [output_targets; ...
        rx_real(n), rx_imag(n)];
end

% Define Neural Network
layers = [
    featureInputLayer(2*M, 'Normalization', 'none', 'Name', 'InputLayer')
    fullyConnectedLayer(hidden_units, 'Name', 'FC1')
    reluLayer('Name', 'ReLU1') % Nonlinear activation
    fullyConnectedLayer(2, 'Name', 'OutputLayer') % Two outputs: real and imaginary
    nmseLoss('NMSELoss')  % 使用自定义的 NMSE 损失层
];

% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', true);

% Train NN
net = trainNetwork(input_features, output_targets, layers, options);

%% Evaluation Data (Remaining 9 Frames)
C_dB_Eval = zeros(1, num_eval_frames);

for frame_idx = 1:num_eval_frames
    eval_idx = frame_idx + num_train_frames;
    txdata_eval = txdata_frames(:, eval_idx);
    linear_residual_eval_norm = linear_residual_frames_norm(:, eval_idx);
    linear_residual_eval = linear_residual_eval_norm * NormalizationFactor; % Rescale for evaluation

    % Extract real and imaginary parts for evaluation
    tx_real_eval = real(txdata_eval);
    tx_imag_eval = imag(txdata_eval);

    % Prepare input features for evaluation
    input_features_eval = [];
    for n = M:frame_size
        input_features_eval = [input_features_eval; ...
            tx_real_eval(n-M+1:n)', tx_imag_eval(n-M+1:n)'];
    end

    % Predict SI using the trained NN
    predicted_output = predict(net, input_features_eval);
    predicted_real = predicted_output(:, 1);
    predicted_imag = predicted_output(:, 2);
    predicted_si_norm = predicted_real + 1j * predicted_imag;
    predicted_si = predicted_si_norm * NormalizationFactor; % Rescale prediction

    % Residual after NN SIC
    residual_after_SIC = linear_residual_eval(M:end) - predicted_si;

    % Calculate SIC Performance
    P_before = sum(abs(linear_residual_eval).^2);
    P_after = sum(abs(residual_after_SIC).^2);
    C_dB_Eval(frame_idx) = 10 * log10(P_before / P_after);
end

%% Calculate and Display Mean and Variance
mean_C_dB = mean(C_dB_Eval);
var_C_dB = var(C_dB_Eval);
disp(['Mean SIC Performance (C in dB): ', num2str(mean_C_dB)]);
disp(['Variance of SIC Performance (C in dB^2): ', num2str(var_C_dB)]);

%% Plot SIC Performance
figure;
plot(1:num_eval_frames, C_dB_Eval, 'b-o', 'LineWidth', 1.5);
xlabel('Frame Index');
ylabel('SIC Performance (C in dB)');
title('SIC Performance Across Evaluation Frames');
grid on;
save('Simplenn200',"C_dB_Eval");
%% PSD Plot for First Frame Using FFT
rxdata_frame = rxdata_frames(:, 1);
analog_residual_frame = analog_residual_frames(:, 1);
linear_residual_frame = linear_residual_frames(:, 1);
nn_residual_frame = residual_after_SIC; % Residual after NN SIC for first frame

N = length(rxdata_frame); % Length of frame

% FFT calculation and normalization
fft_rx = fftshift(fft(rxdata_frame, N));
fft_analog = fftshift(fft(analog_residual_frame, N));
fft_linear = fftshift(fft(linear_residual_frame, N));
fft_nn = fftshift(fft(nn_residual_frame, N));

f = linspace(-fs/2, fs/2, N); % Frequency vector

% Power Spectral Density calculation
psd_rx = 10*log10(abs(fft_rx).^2 / N);
psd_analog = 10*log10(abs(fft_analog).^2 / N);
psd_linear = 10*log10(abs(fft_linear).^2 / N);
psd_nn = 10*log10(abs(fft_nn).^2 / N);

% Smoothing the PSD
psd_rx_smooth = smooth(psd_rx, 20);
psd_analog_smooth = smooth(psd_analog, 20);
psd_linear_smooth = smooth(psd_linear, 20);
psd_nn_smooth = smooth(psd_nn, 20);

% Plotting
figure;
hold on;
plot(f, psd_rx_smooth, 'k', 'LineWidth', 1.5);
plot(f, psd_analog_smooth, 'b', 'LineWidth', 1.5);
plot(f, psd_linear_smooth, 'r', 'LineWidth', 1.5);
plot(f, psd_nn_smooth, 'g', 'LineWidth', 1.5);

xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('PSD of First Frame after Various SIC Stages (FFT and Smooth)');
legend('Original RX', 'After Analog SIC', 'After Linear SIC', 'After NN SIC');
grid on;
hold off;
















