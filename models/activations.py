import numpy as np


class ReLU:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, output_error, learning_rate):
        return output_error * (self.outputs > 0)


class Sigmoid:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, output_error, learning_rate):
        return output_error * self.outputs * (1 - self.outputs)


class Softmax:
    @staticmethod
    def forward(inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
