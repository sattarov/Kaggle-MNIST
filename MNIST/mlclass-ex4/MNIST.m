clear ; close all; clc
f = 'train.csv';
mnist = csvread(f,2);
X = mnist(:,2:end);
y = mnist(:,1);
yCopy = y;
y(yCopy==0)=10;
input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


options = optimset('MaxIter', 100);

lambda = 10;
l = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.3, 2.6, 5.2, 8];
accuracy = [];
for lambda = l
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

fprintf('\nIteration with lambda = %f \n', lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, X);
p = mean(double(pred == y)) * 100;
accuracy = [accuracy p];
fprintf('\nAccuracy = %f\n', p);
fprintf('\nAccuracies = %f\n', accuracy);
end
plot(accuracy);

fprintf('Program paused. Press enter to continue.\n');
pause;

