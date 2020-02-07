function J = computeCost(X, y, theta)

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;




dif=X*theta - y;
J=(dif'*dif)/(2*m);

end
