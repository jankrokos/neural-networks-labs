from utils import create_mini_batches


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, epochs, learning_rate, loss_function, batch_size):
        for epoch in range(epochs):

            mini_batches = create_mini_batches(x_train, y_train, batch_size)

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch

                output = self.forward(x_mini)

                loss = loss_function.forward(output, y_mini)

                output_error = loss_function.backward(output, y_mini)
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, learning_rate)

            if epoch % 250 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {loss:.3f}')
