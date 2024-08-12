import numpy as np

# Network architecture specifications
input_size = 4
hidden_size = 8
output_size = 3

# Initialise weights & biases
w1 = np.ones((input_size, hidden_size))
b1 = np.ones((1, hidden_size))

w2 = np.ones((hidden_size, output_size))
b2 = np.ones((1, output_size))

# Get input and target values from the user
_input = [float(input()) for _ in range(7)]
X = np.array(_input[:4]).reshape(1, -1)
target = np.array(_input[4:]).reshape(1, -1)
learning_rate = 0.1


# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sum-of-squares loss function
def sum_of_squares_loss(output, target):
    return 0.5 * np.sum((output - target) ** 2)


# Feedforward step to compute the output of the network
# Output of the hidden layer using sigmoid activation function
hidden_input = np.dot(X, w1) + b1
hidden_output = sigmoid(hidden_input)

# Output of the output layer using sigmoid activation function
output_input = np.dot(hidden_output, w2) + b2
output = sigmoid(output_input)

# loss before updating weights
loss = sum_of_squares_loss(output, target)

# Implement one iteration of backpropagation and weight update
# Compute gradients
output_delta = (output - target) * output * (1 - output)
hidden_delta = np.dot(output_delta, w2.T) * hidden_output * (1 - hidden_output)

# Update weights and biases
w2 -= learning_rate * np.dot(hidden_output.T, output_delta)
b2 -= learning_rate * (output_delta)
w1 -= learning_rate * np.dot(X.T, hidden_delta)
b1 -= learning_rate * (hidden_delta)

# repeat feedforward of input values into the updated network
# Updated output of the hidden layer
hidden_input_new = np.dot(X, w1) + b1
hidden_output_new = sigmoid(hidden_input_new)

# Updated output of the output layer
output_input_new = np.dot(hidden_output_new, w2) + b2
output_new = sigmoid(output_input_new)

# Compute the new loss value
new_loss = sum_of_squares_loss(output_new, target)

# print("Output of the network for input X:", output)
print(round(loss, 4))

# print("New output after one iteration of backpropagation:", output_new)
print(round(new_loss, 4))