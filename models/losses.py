import numpy as np


class MeanSquaredError:
    @staticmethod
    def forward(predictions, targets):
        return np.mean(np.power(targets - predictions, 2))

    @staticmethod
    def backward(predictions, targets):
        return 2 * (predictions - targets) / targets.size


class CrossEntropyLoss:
    @staticmethod
    def forward(predictions, targets):
        if len(targets.shape) == 1:
            targets = np.eye(predictions.shape[1])[targets]
        return -np.sum(targets * np.log(predictions + 1e-15)) / targets.shape[0]

    @staticmethod
    def backward(predictions, targets):
        if len(targets.shape) == 1:
            targets = np.eye(predictions.shape[1])[targets]
        return predictions - targets
