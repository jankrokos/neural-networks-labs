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
        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        correct_confidences = predictions_clipped[range(samples), targets]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    @staticmethod
    def backward(predictions, targets):
        samples = len(predictions)
        predictions[range(samples), targets] -= 1
        return predictions / samples
