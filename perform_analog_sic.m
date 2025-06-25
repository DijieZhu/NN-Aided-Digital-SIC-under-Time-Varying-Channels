function analog_residual_frame = perform_analog_sic(txdata_frame, rxdata_frame, fs, frame_size)
    % Function to perform Analog SIC on a given frame with fixed channel parameters
    % Inputs:
    %   - txdata_frame: Transmitted data for the frame
    %   - rxdata_frame: Received data for the frame
    %   - fs: Sampling frequency
    %   - frame_size: Frame size (number of samples)
    % Outputs:
    %   - analog_residual_frame: Residual after analog SIC

    % Define fixed channel parameters
    strongest_gain_values = [10^-5.57, 10^-5.77, 10^-6.17];
    strongest_delay_values = [12.5, 25, 37.5]; % in nanoseconds
    [~, strongest_path_index] = max(strongest_gain_values);
    strongest_gain = strongest_gain_values(strongest_path_index);
    strongest_delay_samples = round(strongest_delay_values(strongest_path_index) * fs / 1e9);

    % Create cancellation signal
    cancellation_signal = [zeros(strongest_delay_samples, 1); ...
                           txdata_frame(1:end-strongest_delay_samples)] * strongest_gain;
    cancellation_signal = cancellation_signal(1:frame_size);

    % Perform analog SIC
    analog_residual_frame = rxdata_frame - cancellation_signal;
end
