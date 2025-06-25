clc;
clear;
close all;

%% Parameters
frame_size = 1000;
num_frames = 100;
training_frames = 10;
evaluation_frames = num_frames - training_frames;
M_est = 13;
num_symbols_for_estimation_values = [200, 300, 400]; % Three values for num_symbols_for_estimation
hidden_units = 17;
fs = 20e6;
lambda = 1e-3;
P = 7; % Maximum polynomial order for Polynomial SIC

%% Generate Data for 100 Frames
[txdata_frames, rxdata_frames] = generate_frames(num_frames, frame_size, false);

%% Results storage
poly_sic_results = zeros(length(num_symbols_for_estimation_values), evaluation_frames);
nn_sic_results = zeros(length(num_symbols_for_estimation_values), evaluation_frames);

for est_idx = 1:length(num_symbols_for_estimation_values)
    num_symbols_for_estimation = num_symbols_for_estimation_values(est_idx);

    %% Linear Cancellation (Least Squares)
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

    %% Polynomial SIC
    poly_sic_improvement = zeros(1, num_frames);

    for frame_idx = 1:num_frames
        txdata = txdata_frames(:, frame_idx);
        rxdata = rxdata_frames(:, frame_idx);

        txdata_est = txdata(1:num_symbols_for_estimation);
        rxdata_est = rxdata(1:num_symbols_for_estimation);

        n_features = M_est * P * (P + 1) / 2;
        X_est = zeros(num_symbols_for_estimation, n_features);

        feature_idx = 1;
        for delay = 0:M_est-1
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

        c_mp = (X_est' * X_est + lambda * eye(size(X_est, 2))) \ (X_est' * rxdata_est);

        X_full = zeros(frame_size, n_features);
        feature_idx = 1;
        for delay = 0:M_est-1
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

        si_est_mp = X_full * c_mp;
        rx_clean_mp = rxdata - si_est_mp;

        power_before = mean(abs(rxdata).^2);
        power_after = mean(abs(rx_clean_mp).^2);
        poly_sic_improvement(frame_idx) = 10 * log10(power_before / power_after);
    end

    %% Neural Network SIC
    training_features = [];
    training_targets = [];

    for frame_idx = 1:training_frames
        txdata_train = txdata_frames(:, frame_idx);
        linear_residual_train = linear_residual_frames(:, frame_idx);
        h_ls = channel_coefficients(:, frame_idx);

        tx_real = real(txdata_train);
        tx_imag = imag(txdata_train);

        for n = M_est:frame_size
            x_d_real = tx_real(n-M_est+1:n);
            x_d_imag = tx_imag(n-M_est+1:n);

            f_i = zeros(1, 2 * M_est);
            for i = 1:(2 * M_est)
                idx = n - floor((i - 1) / 2);

                if idx >= 1 && idx <= frame_size
                    h_current = h_ls(ceil(i / 2));
                    x_d_current = txdata_train(idx);

                    if mod(i, 2) == 1
                        f_i(i) = real(h_current * x_d_current);
                    else
                        f_i(i) = imag(h_current * x_d_current);
                    end
                end
            end

            feature_vector = [x_d_real', x_d_imag', f_i];
            training_features = [training_features; feature_vector];
            training_targets = [training_targets; real(linear_residual_train(n)), imag(linear_residual_train(n))];
        end
    end

    layers = [
        featureInputLayer(4*M_est, 'Normalization', 'none', 'Name', 'InputLayer')
        fullyConnectedLayer(hidden_units, 'Name', 'FC1')
        reluLayer('Name', 'ReLU1')
        fullyConnectedLayer(2, 'Name', 'OutputLayer')
        regressionLayer('Name', 'RegressionOutput')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 0.001, ...
        'Verbose', true);

    net = trainNetwork(training_features, training_targets, layers, options);

    nn_sic_improvement = zeros(1, evaluation_frames);

    for frame_idx = 1:evaluation_frames
        eval_idx = frame_idx + training_frames;
        txdata_eval = txdata_frames(:, eval_idx);
        linear_residual_eval = linear_residual_frames(:, eval_idx);
        h_ls = channel_coefficients(:, eval_idx);

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
                end
            end

            feature_vector = [x_d_real', x_d_imag', f_i];
            input_features_eval = [input_features_eval; feature_vector];
        end

        predicted_output = predict(net, input_features_eval);
        predicted_si = predicted_output(:, 1) + 1j * predicted_output(:, 2);

        rx_clean_nn = linear_residual_eval(M_est:end) - predicted_si;
        power_before_nn = mean(abs(linear_residual_eval(M_est:end)).^2);
        power_after_nn = mean(abs(rx_clean_nn).^2);

        nn_sic_improvement(frame_idx) = 10 * log10(power_before_nn / power_after_nn);
    end

    %% Store results
    poly_sic_results(est_idx, :) = poly_sic_improvement(training_frames+1:end);
    nn_sic_results(est_idx, :) = nn_sic_improvement;
end

%% Plot results
for idx = 1:length(num_symbols_for_estimation_values)
    figure;
    hold on;
    plot(poly_sic_results(idx, :), 'b-o', 'DisplayName', 'Polynomial SIC');
    plot(nn_sic_results(idx, :), 'r-s', 'DisplayName', 'NN SIC');
    xlabel('Frame Index');
    ylabel('SIC Performance (dB)');
    title(['SIC Performance (', num2str(num_symbols_for_estimation_values(idx)), ' Symbols)']);
    legend;
    grid on;
end