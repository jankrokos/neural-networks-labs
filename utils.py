import pickle

import numpy as np
from scipy.ndimage import rotate


def create_mini_batches(x, y, batch_size):
    mini_batches = []
    data = np.hstack((x, y))
    np.random.shuffle(data)
    n_mini_batches = data.shape[0] // batch_size

    for i in range(n_mini_batches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        x_mini = mini_batch[:, :-y.shape[1]]
        y_mini = mini_batch[:, -y.shape[1]:]
        mini_batches.append((x_mini, y_mini))

    if data.shape[0] % batch_size != 0:
        mini_batch = data[n_mini_batches * batch_size:data.shape[0]]
        x_mini = mini_batch[:, :-y.shape[1]]
        y_mini = mini_batch[:, -y.shape[1]:]
        mini_batches.append((x_mini, y_mini))

    return mini_batches


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


FASHION_MNIST_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def save_model(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(self, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def flip_image(image):
    image_2d = image.reshape((28, 28))
    flipped_image = np.fliplr(image_2d)

    return flipped_image.reshape((784,))


def rotate_image(image, angle):
    image_2d = image.reshape((28, 28))
    rotated_image = rotate(image_2d, angle, reshape=False, mode='nearest')

    return rotated_image.reshape((784,))


def add_noise(image, noise_level=0.1):
    image_2d = image.reshape((28, 28))
    noisy_image = image_2d + np.random.randn(*image_2d.shape) * noise_level
    noisy_image_clipped = np.clip(noisy_image, 0, 1)

    return noisy_image_clipped.reshape((784,))
