from tensorflow.keras.datasets import fashion_mnist
from numpy.typing import NDArray


def load_fashion_mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test
