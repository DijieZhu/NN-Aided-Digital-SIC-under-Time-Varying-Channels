clc;
clear;
close all;

%% Parameters
frame_size = 1000;
num_frames = 100;
training_frames = 10;
evaluation_frames = num_frames - training_frames;
M_est = 13;
num_symbols_for_estimation = 200;
hidden_units = 17;
fs = 20e6;
lambda = 1e-3;

%% Generate Data for 100 Frames
[txdata_frames, rxdata_frames] = generate_frames(num_frames, frame_size, false);



%% Linear Cancellation (Least Squares with 400 Symbols)
linear_residual_frames = zeros(size(rxdata_frames));
channel_coefficients = zeros(M_est, num_frames);

for frame_idx = 1:num_frames
    txdata_frame = txdata_frames(1:num_symbols_for_estimation, frame_idx);
    rxdata_frame = rxdata_frames(1:num_symbols_for_estimation, frame_idx);

    num_valid_samples = num_symbols_for_estimation - M_est;
    X = zeros(num_valid_samples, M_est);
    y = rxdata_frame(M_est+1:num_symbols_for_estimation);

    for i = 1:num_valid_samples
        n = i + M_est;
        X(i, :) = txdata_frame(n-M_est+1:n).';
    end

    h_ls = (X' * X + lambda * eye(size(X' * X))) \ (X' * y);
    channel_coefficients(:, frame_idx) = h_ls;

    full_X = zeros(frame_size - M_est, M_est);
    for i = 1:frame_size - M_est
        n = i + M_est;
        full_X(i, :) = txdata_frames(n-M_est+1:n, frame_idx).';
    end

    linear_component = zeros(frame_size, 1);
    linear_component(M_est+1:frame_size) = full_X * h_ls;
    linear_residual_frames(:, frame_idx) = rxdata_frames(:, frame_idx) - linear_component;
end

%% Normalize for NN Training
% NormalizationFactor = 1;
NormalizationFactor = max(max(abs(linear_residual_frames)));
linear_residual_frames_norm = linear_residual_frames / NormalizationFactor;

%% Training Data Preparation (First 10 Frames)
training_features = [];
training_targets = [];

for frame_idx = 1:training_frames
    txdata_train = txdata_frames(:, frame_idx);
    linear_residual_train = linear_residual_frames_norm(:, frame_idx);
    h_ls = channel_coefficients(:, frame_idx);

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
                % Get the corresponding channel coefficient
                h_current = h_ls(ceil(i / 2));
                x_d_current = txdata_train(idx);

                % Calculate f_i based on whether i is odd or even
                if mod(i, 2) == 1  % i is odd -> real part
                    f_i(i) = real(h_current * x_d_current);
                else  % i is even -> imaginary part
                    f_i(i) = imag(h_current * x_d_current);
                end
            else
                error('Sliding window index out of bounds: idx = %d', idx);
            end
        end
        % Combining all features: [x_d_real, x_d_imag, f_i]
        feature_vector = [x_d_real', x_d_imag', f_i];
        
        % Append feature vector and target to training data
        training_features = [training_features; feature_vector];
        training_targets = [training_targets; real(linear_residual_train(n)), imag(linear_residual_train(n))];
    end
end


%% Neural Network Configuration
layers = [
    featureInputLayer(4*M_est, 'Normalization', 'none', 'Name', 'InputLayer')
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

%% Evaluation (Remaining 90 Frames)
C_dB_Total = zeros(1, evaluation_frames);

for frame_idx = 1:evaluation_frames
    eval_idx = frame_idx + training_frames;
    txdata_eval = txdata_frames(:, eval_idx);
    linear_residual_eval_norm = linear_residual_frames_norm(:, eval_idx);
    h_ls = channel_coefficients(:, eval_idx);

    tx_real_eval = real(txdata_eval);
    tx_imag_eval = imag(txdata_eval);

    input_features_eval = [];

    for n = M_est:frame_size
        % Historical signal real and imaginary parts
        x_d_real = tx_real_eval(n-M_est+1:n);
        x_d_imag = tx_imag_eval(n-M_est+1:n);

        % Calculating f_i as per the given definition
        f_i = zeros(1, 2 * M_est);
        for i = 1:(2 * M_est)
            idx = n - floor((i - 1) / 2);

            if idx >= 1 && idx <= frame_size
                % Get the corresponding channel coefficient
                h_current = h_ls(ceil(i / 2));
                x_d_current = txdata_eval(idx);

                % Calculate f_i based on whether i is odd or even
                if mod(i, 2) == 1  % i is odd -> real part
                    f_i(i) = real(h_current * x_d_current);
                else  % i is even -> imaginary part
                    f_i(i) = imag(h_current * x_d_current);
                end
            else
                error('Sliding window index out of bounds: idx = %d', idx);
            end
        end

        % Combining all features: [x_d_real, x_d_imag, f_i]
        feature_vector = [x_d_real', x_d_imag', f_i];
        
        % Append feature vector to the evaluation set
        input_features_eval = [input_features_eval; feature_vector];
    end

    % Predict SI using the trained NN
    predicted_output = predict(net, input_features_eval);
    predicted_real = predicted_output(:, 1);
    predicted_imag = predicted_output(:, 2);
    predicted_si_norm = predicted_real + 1j * predicted_imag;
    predicted_si = predicted_si_norm * NormalizationFactor;

    % Residual after SIC
    linear_residual_current_frame = linear_residual_frames(:, eval_idx);
    residual_after_SIC = linear_residual_current_frame(M_est:end) - predicted_si;

    % Calculate SIC Performance
    P_before = sum(abs(linear_residual_frames(:, eval_idx)).^2);
    P_after = sum(abs(residual_after_SIC).^2);
    C_dB_Total(frame_idx) = 10 * log10(P_before / P_after);
end


%% Plot SIC Performance
figure;
plot(1:evaluation_frames, C_dB_Total, 'b-o', 'LineWidth', 1.5);
xlabel('Frame Index');
ylabel('SIC Performance (C in dB)');
title('SIC Performance Across Frames (90 Evaluation Frames)');
xlim([1, num_frames]);
ylim([-20, 30]);
grid on;

mean_C_dB = mean(C_dB_Total);
var_C_dB = var(C_dB_Total);
disp(['Mean SIC Performance (C in dB): ', num2str(mean_C_dB)]);
disp(['Variance of SIC Performance (C in dB^2): ', num2str(var_C_dB)]);


%% PSD Plot for the First Frame
frame_idx = 1; % First frame
nfft = 4096; % Number of FFT points

% Align predicted_si with linear_residual_frames
aligned_predicted_si = zeros(frame_size, 1);
aligned_predicted_si(M_est:end) = predicted_si; % Fill aligned array starting from M_est

% Compute PSDs
rxdata_psd = 10 * log10(abs(fftshift(fft(rxdata_frames(:, frame_idx), nfft))).^2 / nfft);
analog_sic_psd = 10 * log10(abs(fftshift(fft(rxdata_frames(:, frame_idx) - linear_residual_frames(:, frame_idx), nfft))).^2 / nfft);
linear_sic_psd = 10 * log10(abs(fftshift(fft(linear_residual_frames(:, frame_idx), nfft))).^2 / nfft);
nn_sic_psd = 10 * log10(abs(fftshift(fft(linear_residual_frames(:, frame_idx) - aligned_predicted_si, nfft))).^2 / nfft);

% Frequency axis
fs = 20e6; % Sampling frequency
freq_axis = linspace(-fs/2, fs/2, nfft);

% Smooth the PSDs
window_size = 10; % Moving average window size
smooth_rxdata_psd = smooth(rxdata_psd, window_size);
smooth_analog_sic_psd = smooth(analog_sic_psd, window_size);
smooth_linear_sic_psd = smooth(linear_sic_psd, window_size);
smooth_nn_sic_psd = smooth(nn_sic_psd, window_size);

% Plot PSDs
figure;
plot(freq_axis / 1e6, smooth_rxdata_psd, 'b', 'LineWidth', 1.5); hold on;
plot(freq_axis / 1e6, smooth_analog_sic_psd, 'r', 'LineWidth', 1.5);
plot(freq_axis / 1e6, smooth_linear_sic_psd, 'g', 'LineWidth', 1.5);
plot(freq_axis / 1e6, smooth_nn_sic_psd, 'm', 'LineWidth', 1.5);
hold off;

% Add labels, legend, and title
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB)');
title('PSD Comparison for First Frame');
legend('Received Signal', 'Analog SIC', 'Linear SIC', 'NN SIC');
grid on;






