import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size, activation_fn=None):
        self.activation_fn = activation_fn

        self.inputs = None
        self.outputs = None

        # Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))

        # # Random initialization
        # self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        x = np.array(x, ndmin=2)
        self.inputs = x
        self.outputs = np.dot(x, self.weights) + self.biases
        if self.activation_fn:
            self.outputs = self.activation_fn.forward(self.outputs)
        return self.outputs

    def backward(self, output_error, learning_rate):
        d_activation = self.activation_fn.backward(output_error, learning_rate) if self.activation_fn else output_error

        input_error = np.dot(d_activation, self.weights.T)

        weights_error = np.dot(self.inputs.T, d_activation)
        biases_error = np.sum(d_activation, axis=0, keepdims=True)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biases_error

        return input_error
