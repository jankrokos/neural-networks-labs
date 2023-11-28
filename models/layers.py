import numpy as np

from models.activations import ReLU
from models.initializers import random_init, xavier_init, he_init


class DenseLayer:
    def __init__(self, input_size, output_size, activation_fn=None, initializer=random_init, l2_reg=0.0):
        self.activation_fn = activation_fn
        self.l2_reg = l2_reg

        self.inputs = None
        self.outputs = None

        self.weights = initializer(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        x = np.array(x, ndmin=2)
        self.inputs = x
        self.outputs = np.dot(x, self.weights) + self.biases
        if self.activation_fn:
            self.outputs = self.activation_fn.forward(self.outputs)
        return self.outputs

    def backward(self, output_error, learning_rate):
        if self.activation_fn and hasattr(self.activation_fn, 'backward'):
            d_activation = self.activation_fn.backward(output_error, learning_rate)
        else:
            d_activation = output_error

        input_error = np.dot(d_activation, self.weights.T)

        weights_error = np.dot(self.inputs.T, d_activation)
        biases_error = np.sum(d_activation, axis=0, keepdims=True)

        l2_penalty = 0
        if self.l2_reg > 0:
            l2_penalty = self.l2_reg * np.sum(self.weights ** 2)
            weights_error += l2_penalty * self.weights

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biases_error

        return input_error, l2_penalty


class FlattenLayer:
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.original_shape), 0


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initializes the Convolutional Layer.

        Parameters:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of filters, determining the number of output channels.
        kernel_size (int): Size of the filter/kernel (assumed to be a square).
        stride (int): The stride of the convolution operation. Defaults to 1.
        padding (int): The amount of zero-padding added to both sides of the input. Defaults to 0.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization for weights - good for layers before ReLU activation
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2. / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        """
        Performs the forward pass of the convolutional layer.

        Parameters:
        x (numpy.ndarray): Input data of shape (batch_size, in_channels, height, width).

        Returns:
        numpy.ndarray: Output data after applying the convolution operation.
        """
        # Padding the input to control the output size
        self.inputs = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                             mode='constant')

        batch_size, in_channels, in_height, in_width = self.inputs.shape
        out_height = int((in_height - self.kernel_size) / self.stride + 1)
        out_width = int((in_width - self.kernel_size) / self.stride + 1)

        self.outputs = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Apply the convolution operation
        for i in range(out_height):
            for j in range(out_width):
                y_slice = i * self.stride
                x_slice = j * self.stride
                input_slice = self.inputs[:, :, y_slice:y_slice + self.kernel_size, x_slice:x_slice + self.kernel_size]
                for k in range(self.out_channels):
                    self.outputs[:, k, i, j] = np.sum(input_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + \
                                               self.biases[k]

        self.outputs = np.maximum(0, self.outputs)  # ReLU
        return self.outputs

    def backward(self, output_error, learning_rate):
        """
        Performs the backward pass of the convolutional layer.

        Parameters:
        output_error (numpy.ndarray): Gradient of the loss function with respect to the output of this layer.
        learning_rate (float): Learning rate for weight update.

        Returns:
        Tuple[numpy.ndarray, float]: Gradient of the loss function with respect to the input of this layer,
        and the L2 regularization penalty (which is 0 for this layer).
        """
        output_error = output_error * (self.outputs > 0)  # Gradient of ReLU activation
        batch_size, in_channels, in_height, in_width = self.inputs.shape
        out_height, out_width = output_error.shape[2], output_error.shape[3]

        # Initialize gradients with respect to weights, biases, and input
        weights_error = np.zeros(self.weights.shape)
        biases_error = np.zeros(self.biases.shape)
        input_error = np.zeros(self.inputs.shape)

        # Calculate gradients
        for y in range(out_height):
            for x in range(out_width):
                y_slice = y * self.stride
                x_slice = x * self.stride
                input_slice = self.inputs[:, :, y_slice:y_slice + self.kernel_size, x_slice:x_slice + self.kernel_size]

                for k in range(self.out_channels):
                    weights_error[k, :, :, :] += np.sum(input_slice * output_error[:, k:k + 1, y:y + 1, x:x + 1],
                                                        axis=0)
                    biases_error[k] += np.sum(output_error[:, k, y, x], axis=0)
                    input_error[:, :, y_slice:y_slice + self.kernel_size, x_slice:x_slice + self.kernel_size] += \
                        self.weights[k, :, :, :] * output_error[:, k:k + 1, y:y + 1, x:x + 1]

        # Update weights and biases
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biases_error

        # Return the input error for further backpropagation and the L2 penalty (0 for this layer)
        return input_error[:, :, self.padding:in_height - self.padding, self.padding:in_width - self.padding], 0


class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        """
        Initializes the MaxPooling Layer.

        Parameters:
        pool_size (int): The size of the window to take the max over. Defaults to 2.
        stride (int): The stride of the pooling operation. Defaults to 2.
        """
        self.pool_size = pool_size
        self.stride = stride
        self.inputs = None
        self.outputs = None

    def forward(self, x):
        """
        Performs the forward pass of the max pooling layer.

        Parameters:
        x (numpy.ndarray): Input data of shape (batch_size, channels, height, width).

        Returns:
        numpy.ndarray: Output data after applying max pooling.
        """
        self.inputs = x
        batch_size, channels, height, width = x.shape

        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        # Initialize output
        self.outputs = np.zeros((batch_size, channels, out_height, out_width))

        # Apply max pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # Select window and apply pooling
                window = x[:, :, h_start:h_end, w_start:w_end]
                self.outputs[:, :, i, j] = np.max(window, axis=(2, 3))

        return self.outputs

    def backward(self, output_error, learning_rate):
        """
        Performs the backward pass of the max pooling layer.

        Parameters:
        output_error (numpy.ndarray): Gradient of the loss function with respect to the output of this layer.
        learning_rate (float): Learning rate for the layer. Not used in this layer as it has no learnable parameters.

        Returns:
        numpy.ndarray: Gradient of the loss function with respect to the input of this layer.
        """
        batch_size, channels, height, width = self.inputs.shape
        out_height, out_width = output_error.shape[2], output_error.shape[3]

        # Initialize input gradient
        input_error = np.zeros_like(self.inputs)

        # Calculate gradient
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # Calculate gradient for each window
                for b in range(batch_size):
                    for c in range(channels):
                        window = self.inputs[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        input_error[b, c, h_start:h_end, w_start:w_end] += output_error[b, c, i, j] * mask

        return input_error, 0
