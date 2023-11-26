import numpy as np

from utils import create_mini_batches


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.losses = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, epochs, learning_rate, loss_function, batch_size):
        loss = None
        for epoch in range(epochs):

            mini_batches = create_mini_batches(x_train, y_train, batch_size)

            total_l2_penalty = 0
            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch

                output = self.forward(x_mini)

                loss = loss_function.forward(output, y_mini)

                output_error = loss_function.backward(output, y_mini)
                for layer in reversed(self.layers):
                    output_error, l2_penalty = layer.backward(output_error, learning_rate)
                    total_l2_penalty += l2_penalty

            loss += total_l2_penalty / (2 * x_train.shape[0])
            print(f'Epoch: {epoch + 1}, Loss: {loss:.3f}')
            self.losses.append(loss)

    def evaluate(self, x_test, y_test):
        output = self.forward(x_test)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == labels)
        print(f'Accuracy: {(accuracy * 100):.2f}%')
