function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad_1 = zeros(size(theta));
grad_2 = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypo=theta'*X';
hypo=hypo';
sigmoid=1./(1+e.^(-hypo));
cost_fun=y.*log(sigmoid)+(1-y).*log(1-sigmoid);
o=ones(1,m);
j1=-o*cost_fun/m;
theta_c=theta(2:end,:);
j2=lambda*((ones(size(theta_c)))'*(theta_c.^2))/(2.*m);
J=j1+j2;

% gradient
grad_1(1,1)=(ones(1,m)*((sigmoid-y).*X(:,1)));

for i=2:length(theta)
  grad_1(i,1)=ones(1,m)*((sigmoid-y).*X(:,i));
  grad_2(i,1)=lambda*theta(i)/m;
  
endfor
grad=(grad_1/m)+grad_2;





% =============================================================

end
