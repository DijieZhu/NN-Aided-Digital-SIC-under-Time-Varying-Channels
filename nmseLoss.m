classdef nmseLoss < nnet.layer.RegressionLayer
    % NMSELOSS Custom loss layer for Normalized Mean Squared Error

    methods
        function layer = nmseLoss(name)
            % Initialize the layer with the specified name
            layer.Name = name;
            layer.Description = 'Normalized Mean Squared Error';
        end

        function loss = forwardLoss(layer, Y, T)
            % Y - Predictions
            % T - Targets

            % Compute Mean Squared Error
            mse = mean((Y - T).^2, 'all');

            % Compute Normalization factor (e.g., mean of squared targets)
            normFactor = mean(T.^2, 'all');  % 修正：添加了缺失的单引号

            % Compute NMSE
            loss = mse / normFactor;
        end

        function dLdY = backwardLoss(layer, Y, T)
            % Compute gradients for backpropagation

            % Number of elements
            N = numel(Y);

            % Compute normalization factor
            normFactor = mean(T.^2, 'all');  % 修正：添加了缺失的单引号

            % Gradient of MSE with respect to Y
            dMSE_dY = 2 * (Y - T) / N;

            % Gradient of NMSE with respect to Y
            dLdY = dMSE_dY / normFactor;
        end
    end
end
