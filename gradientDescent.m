function [theta, J_history, theta_history] = gradientDescent(X, Y, theta, alpha, num_iters)

% Initialize some useful values
m = length(Y); % number of training examples
J_history = zeros(num_iters+1, 1);
theta_history = zeros(num_iters+1, size(theta', 2));
theta_history(1, :)=theta';
J_history(1)= computeCost(X, Y, theta);


for iter = 2:(num_iters+1)

   
    theta_prev = theta;

    % number of features.
    p = size(X, 2);

    % simultaneous update theta using theta_prev.
    for j = 1:p

       
        deriv = ((X*theta_prev - Y)'*X(:, j))/m;

        % update theta_j
        theta(j) = theta_prev(j)-(alpha*deriv);
    end
    %
    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, Y, theta);
    theta_history(iter, :) = theta';
end

end