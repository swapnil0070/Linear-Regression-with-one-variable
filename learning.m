# Created by Octave 5.1.0, Sat Feb 08 00:18:56 2020 GMT <unknown@BT1906519>
fprintf('Plotting Data ...\n')
data = csvread('newhousing.txt');
x = data(1:16000, 3); Y = data(1:16000, 9);%Taking 80 percent of the data set to train the model
X=x./10;
y=Y./100000;
m = length(Y); % number of training examples


plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), X(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);


% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);

J = computeCost(X, y, [-1 ; 0]);
fprintf('\nWith theta = [-1 ; 0]\nCost computed = %f\n', J);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);


% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-r',10)
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for House age
predict1 = [1, 3.5] *theta;
fprintf('For house age=35 years, we predict a house value of %f\n',...
    predict1*100000);
predict2 = [1, 7] * theta;
fprintf('For population = 70 years, we predict a house value of %f\n',...
    predict2*100000);

fprintf('Program paused. Press enter to continue.\n');
pause;