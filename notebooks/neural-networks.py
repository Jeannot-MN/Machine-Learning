import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases with ones
        self.weights = {
            'hidden': np.ones((4, 8)),
            'output': np.ones((8, 3))
        }
        self.biases = {
            'hidden': np.ones((1, 8)),
            'output': np.ones((1, 3))
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        self.input_layer = inputs
        self.hidden_layer = self.sigmoid(np.dot(self.input_layer, self.weights['hidden']) + self.biases['hidden'])
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights['output']) + self.biases['output'])
        return self.output_layer

    def compute_loss(self, output, target):
        # Compute the sum-of-squares loss
        loss = 0.5 * np.sum((output - target) ** 2)
        return loss

    def backpropagate(self, target, learning_rate=0.1):
        # Compute the error
        output_error = self.output_layer - target
        output_delta = output_error * self.sigmoid_derivative(self.output_layer)

        hidden_error = output_delta.dot(self.weights['output'].T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)

        # Update the weights and biases
        self.weights['output'] -= learning_rate * self.hidden_layer.T.dot(output_delta)
        self.weights['hidden'] -= learning_rate * self.input_layer.T.dot(hidden_delta)

        self.biases['output'] -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.biases['hidden'] -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)


def main():
    nn = NeuralNetwork()

    numbers = []
    for _ in range(7):
        numbers.append(float(input()))

    inputs = np.array([numbers[:4]])
    target = np.array([numbers[4:]])

    # Feedforward and compute loss before training
    output_before = nn.feedforward(inputs)
    loss_before = nn.compute_loss(output_before, target)

    # Backpropagate
    nn.backpropagate(target)

    # Feedforward and compute loss after training
    output_after = nn.feedforward(inputs)
    loss_after = nn.compute_loss(output_after, target)

    # Output the loss values
    print(round(loss_before, 4))
    print(round(loss_after, 4))


main()