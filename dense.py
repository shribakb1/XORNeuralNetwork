import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        # dE/dW = dE/dY * X^T
        weights_gradient = np.dot(output_gradient, self.input.T)
        # dE/dX = W^T * dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)
        # update
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
