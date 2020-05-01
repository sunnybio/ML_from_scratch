function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
 h=theta'.*X;
 h=h(:,1)+h(:,2);
 a=((h-y).^2);
 on=ones(1,m);
 cost=(on*a)./(2*m);
 J=cost;
 




% =========================================================================

end
