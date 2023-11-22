import numpy as np


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
