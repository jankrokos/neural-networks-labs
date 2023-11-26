from tensorflow.keras.datasets import fashion_mnist, mnist
from numpy.typing import NDArray
import numpy as np


def load_fashion_mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test


def load_clothing(label) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    train_filter = np.where(y_train == label)
    test_filter = np.where(y_test == label)

    return x_train[train_filter], y_train[train_filter], x_test[test_filter], y_test[test_filter]


def load_mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test
