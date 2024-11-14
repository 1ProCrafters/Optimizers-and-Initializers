import numpy as np
from scipy import stats
import json

# Function to calculate the confidence interval for initializing bias
def calculate_confidence_interval(data, confidence_level):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    ci_min = mean - (z_score * (std_dev / np.sqrt(n)))
    ci_max = mean + (z_score * (std_dev / np.sqrt(n)))

    return ci_min, ci_max

# Initialize the neural network output bias with confidence interval of target data
def initialize_neural_network(data, confidence_level):
    ci_min, ci_max = calculate_confidence_interval(data, confidence_level)
    initializer = (ci_min + ci_max) / 2
    return initializer

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def derivative_relu(x):
    return (x > 0).astype(int)

def derivative_sigmoid(x):
    return x * (1 - x)

# Neural network function with forward pass
def neural_network(x, weights, biases):
    hidden_layer = sigmoid(np.dot(x, weights['hidden']) + biases['hidden'])
    output_layer = relu(np.dot(hidden_layer, weights['output']) + biases['output'])
    return output_layer

# Training function for the neural network
def train_neural_network(x, y, initializer, learning_rate, num_iterations):
    weights = {
        'hidden': np.random.rand(x.shape[1], 64).tolist(),
        'output': np.random.rand(64, 1).tolist()
    }
    biases = {
        'hidden': np.zeros((1, 64)).tolist(),
        'output': np.full((1, 1), initializer).tolist()
    }

    for i in range(num_iterations):
        # Forward pass
        hidden_layer = sigmoid(np.dot(x, np.array(weights['hidden'])) + np.array(biases['hidden']))
        output_layer = relu(np.dot(hidden_layer, np.array(weights['output'])) + np.array(biases['output']))

        # Calculate error
        output_error = y - output_layer
        output_delta = output_error * derivative_relu(output_layer)
        hidden_error = output_delta.dot(np.array(weights['output']).T)
        hidden_delta = hidden_error * derivative_sigmoid(hidden_layer)

        # Update weights and biases
        weights['output'] = (np.array(weights['output']) + learning_rate * hidden_layer.T.dot(output_delta)).tolist()
        weights['hidden'] = (np.array(weights['hidden']) + learning_rate * x.T.dot(hidden_delta)).tolist()
        biases['output'] = (np.array(biases['output']) + learning_rate * np.sum(output_delta, axis=0, keepdims=True)).tolist()
        biases['hidden'] = (np.array(biases['hidden']) + learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)).tolist()

    return weights, biases

# Function to save the weights and biases to a file
def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

# Example usage
if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.rand(100, 10)
    y = np.random.rand(100, 1)

    confidence_level = 0.95
    initializer = initialize_neural_network(y, confidence_level)

    learning_rate = 0.01
    num_iterations = 1000
    weights, biases = train_neural_network(x, y, initializer, learning_rate, num_iterations)

    # Save weights and biases to a JSON file
    save_to_file({'weights': weights, 'biases': biases}, 'trained_network.json')
    print("Weights and biases have been saved to 'trained_network.json'")

