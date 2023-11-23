import numpy as np


def xavier_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))


def random_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.01


def he_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
