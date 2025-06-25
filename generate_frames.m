function [txdata_frames, rxdata_frames] = generate_frames(num_frames, frame_size, static_channel)
    % Parameters
    M = 64; % 64-QAM modulation
    bits_per_symbol = log2(M);
    sps = 4; % Samples per symbol
    roll_off = 0.25;
    span = 10;
    rc_filter = rcosdesign(roll_off, span, sps, 'sqrt'); % RC filter design
    fs = 20e6; % Sampling Frequency
    fc = 3e9; % Carrier Frequency
    velocity = 10; % Relative speed (km/h)
    c = 3e8; % Speed of light
    v = velocity / 3.6; % Speed in m/s
    fd = v * fc / c; % Max Doppler frequency
    
    % Channel parameters (invariant paths)
    direct_delays = [12.5, 25, 37.5]; % Delays in ns
    direct_sample_delays = round(direct_delays * fs / 1e9); % Delays in samples
    direct_gains_dB = [-55.7, -57.7, -61.7];
    direct_gains_linear = 10 .^ (direct_gains_dB / 10);

    % Channel parameters (varying paths)
    indirect_delays = [50, 62.5, 75, 100, 137.5];
    indirect_sample_delays = round(indirect_delays * fs / 1e9);
    indirect_gains_dB = [-66.2, -76.2, -96.2, -106.2, -112.2];
    indirect_gains_linear = 10 .^ (indirect_gains_dB / 10);
    N_scatterers = 8;
    
    % Initialize output storage
    txdata_frames = zeros(frame_size, num_frames);
    rxdata_frames = zeros(frame_size, num_frames);
    txdata_pa_frames = zeros(frame_size, num_frames);

    % Generate and process each frame independently
    for frame_idx = 1:num_frames
        % Generate random bit data for this frame
        num_of_bits = frame_size * bits_per_symbol;
        bitdata = randi([0, 1], num_of_bits, 1);
        bitgroups = reshape(bitdata, bits_per_symbol, []).';
        symbols = bi2de(bitgroups, 'left-msb');
        txdata = qammod(symbols, M, 'UnitAveragePower', true, 'InputType', 'integer');
        
        % Pulse shaping
        txdata_upsampled = upsample(txdata, sps);
        txdata_shaped = conv(txdata_upsampled, rc_filter, 'same');
        
        % Nonlinearity of PA
        txdata_pa = apply_pa_nonlinearity(txdata_shaped);

        % Generate static or time-varying channel coefficients
        if static_channel
            % Static channel: generate once and reuse
            if frame_idx == 1
                time = (0:frame_size*sps-1)' / fs;
                indirect_path_coeffs_static = generate_channel_coeffs(true, indirect_gains_linear, fd, time, N_scatterers);
                direct_path_coeffs_static = direct_gains_linear; % Fixed direct path coefficients
            end
            indirect_path_coeffs = indirect_path_coeffs_static;
            direct_path_coeffs = direct_path_coeffs_static;
        else
            % Generate time-varying channel coefficients
            time = (0:frame_size*sps-1)' / fs;
            indirect_path_coeffs = generate_channel_coeffs(false, indirect_gains_linear, fd, time, N_scatterers);
            direct_path_coeffs = direct_gains_linear; % Direct path still fixed
        end

        % Apply channel to the PA-processed data
        y_channel = apply_channel(txdata_pa, direct_path_coeffs, direct_sample_delays, indirect_path_coeffs, indirect_sample_delays);
        
        % Add noise (if needed)
        snr = 30;
        rxdata = awgn(y_channel, snr, 'measured');

        % Store frame data
        txdata_frames(:, frame_idx) = txdata_shaped(1:frame_size);
        txdata_pa_frames(:, frame_idx) = txdata_pa(1:frame_size);
        rxdata_frames(:, frame_idx) = rxdata(1:frame_size);
    end
end

function txdata_pa = apply_pa_nonlinearity(txdata_shaped)
    c = [1.0513+0.0904j, -0.0680-0.0023j, 0.0289-0.0054j;
         -0.0542-0.2900j, 0.2234+0.2317j, -0.0621-0.0932j;
         -0.9657-0.7028j, -0.2451-0.3735j, 0.1229+0.1508j];
    x = txdata_shaped;
    y = zeros(size(x));
    for n = 5:length(x)
        for i = 1:2:5 % Odd-order terms only
            for j = 1:3 % Memory terms
                y(n) = y(n) + c((i + 1) / 2, j) * x(n - i + 1) * abs(x(n - i + 1))^(i - 1);
            end
        end
    end
    txdata_pa = y;
end

function indirect_path_coeffs = generate_channel_coeffs(static_channel, gains_linear, fd, time, N_scatterers)
    num_paths = length(gains_linear);
    indirect_path_coeffs = zeros(length(time), num_paths);
    for j = 1:num_paths
        if static_channel
            % Static channel: generate once and reuse
            if j == 1
                phi_static = 2 * pi * rand(N_scatterers, 1);
                f_n_static = fd * cos(2 * pi * (1:N_scatterers)' / N_scatterers);
                h_static = sum(sqrt(1 / N_scatterers) * exp(1j * (2 * pi * f_n_static .* time' + phi_static)), 1).';
            end
            h = h_static;
        else
            % Time-varying channel: regenerate for each frame
            phi = 2 * pi * rand(N_scatterers, 1);
            f_n = fd * cos(2 * pi * (1:N_scatterers)' / N_scatterers);
            h = sum(sqrt(1 / N_scatterers) * exp(1j * (2 * pi * f_n .* time' + phi)), 1).';
        end
        indirect_path_coeffs(:, j) = gains_linear(j) * h;
    end
end

function y_channel = apply_channel(txdata_pa, direct_gains_linear, direct_sample_delays, indirect_path_coeffs, indirect_sample_delays)
    num_time_invariant = length(direct_gains_linear);
    num_time_varying = size(indirect_path_coeffs, 2);
    total_samples = length(txdata_pa);
    y_channel = zeros(total_samples, 1);
    for i = 1:num_time_invariant
        delayed_signal = [zeros(direct_sample_delays(i), 1); txdata_pa(1:end-direct_sample_delays(i))];
        delayed_signal = delayed_signal(1:total_samples);
        y_channel = y_channel + direct_gains_linear(i) * delayed_signal;
    end
    for j = 1:num_time_varying
        delayed_signal = [zeros(indirect_sample_delays(j), 1); txdata_pa(1:end-indirect_sample_delays(j))];
        delayed_signal = delayed_signal(1:total_samples);
        y_channel = y_channel + indirect_path_coeffs(:, j) .* delayed_signal;
    end
end



