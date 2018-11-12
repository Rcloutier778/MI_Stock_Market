function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% Modification of Andrew Ng Machine Learning Course nnCostFunction

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
n = size(X, 1);  %num training samples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X1 = [ones(n, 1) X];
midLayer = sigmoid(Theta1*X1');
midLayer1 = [ones(1,size(midLayer,2)) ; midLayer];
out = sigmoid(Theta2*midLayer1);
% [maxVal,maxIndex] = max(out',[],2);
% p=maxIndex;

ynn = zeros(n,num_labels);
for i=1:n
    ynn(i,y(i)) = 1;
end
ynn=ynn';

% cost function- first without regularization
% <<update next line>>:

J = (-1/n)*sum(sum(ynn.*log(out) + (1-ynn).*log(1-out)));

% Add regularization to cost function
% <<update next line>>:
% reg = 
reg=(lambda/(2*n))*(sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
J = J+reg;

%backpropogation, calculation of gradients
for t = 1:n
%     % Step 1)  Forward propagate
    a1 = X1(t,:);	% we did the bias above
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [1; a2];	% need to add the bias back to this one
    z3 =  Theta2 * a2;
    a3 = sigmoid(z3);
    z2 = [ 1 ; z2 ];	% still need to worry about the bias effect
% 
%     % Step 2)  Compute error
    deltaPart3 = a3 - ynn(:,t);
%     
%     % Step 3) Back propagate error through activation function
    deltaPart2 = (Theta2' * deltaPart3) .* sigmoidGradient( z2 );
% 
%     % Step 4)  Update weights
    Theta2_grad = Theta2_grad + deltaPart3 * a2';
    Theta1_grad = Theta1_grad + deltaPart2(2:end) * a1;  	
end;
	
% Step 5)  Average gradient update
Theta1_grad = Theta1_grad ./ n;
Theta2_grad = Theta2_grad ./ n;



%regularization
Theta1_tmp = Theta1;
Theta1_tmp(:,1) = 0;	%	don't regularize bias terms
Theta1_grad = Theta1_grad + lambda * Theta1_tmp / n;
Theta2_tmp = Theta2;
Theta2_tmp(:,1) = 0;
Theta2_grad = Theta2_grad + lambda * Theta2_tmp / n;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
