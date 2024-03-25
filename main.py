import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, activation='sigmoid', learning_rate=0.1):
        # Layer 1 (hidden layer) weights and biases
        self.W1 = np.ones((hidden_layer_size, input_size))
        self.B1 = np.ones((hidden_layer_size, 1))

        # Layer 2 (output layer) weights and biases
        self.W2 = np.ones((output_size, hidden_layer_size))
        self.B2 = np.ones((output_size, 1))

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Invalid activation function")
        self.activation = activation

        self.learning_rate = learning_rate

    def feed_forward(self, _input):
        # Layer 1 (hidden layer)
        self.Z1 = np.dot(self.W1, _input) + self.B1
        self.A1 = self.compute_activation_value(self.Z1)

        # Layer 2 (output layer)
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.compute_activation_value(self.Z2)

        return self.A2

    def back_propagate(self, _input, target):
        # Layer 2 (output layer) Deltas
        D2 = (self.A2 - target) * self.A2 * (1 - self.A2)

        # Layer 2 (output layer) Deltas
        D1 = np.dot(self.W2.transpose(), D2) * self.A1 * (1 - self.A1)

        self.W2 -= self.learning_rate * np.dot(D2, self.A1.transpose())
        self.B2 -= self.learning_rate * D2

        self.W1 -= self.learning_rate * np.dot(D1, _input.transpose())
        self.B1 -= self.learning_rate * D1

    def compute_activation_value(self, z):
        activation_output = 0
        if self.activation == 'sigmoid':
            activation_output = 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            activation_output = max(0, z)
        elif self.activation == 'tanh':
            activation_output = np.tanh(z)

        return activation_output

    def compute_loss(self, target):
        return np.sum(np.square(self.A2 - target)) / 2


net = NeuralNetwork(4, 8, 3)
std_input = [float(input()) for _ in range(7)]

_input = np.array(std_input[:4])[:, np.newaxis]
target = np.array(std_input[4:])[:, np.newaxis]

net.feed_forward(_input)
print(round(net.compute_loss(target), 4))

net.back_propagate(_input, target)

net.feed_forward(_input)
print(round(net.compute_loss(target), 4))
