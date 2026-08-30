import numpy as np

weights = np.array([
    [0.2, 0.8, -0.5],
    [0.5, -0.3, 0.7],
    [-0.1, 0.4, 0.9],
    [0.6, 0.2, -0.2],
    [-0.7, 0.3, 0.5]
])
inputs = np.array([
    [1, 2, 3],
    [3, 2, 1],
    [0, 1, 2],
    [4, 3, 2],
    [-1, 0, 1]
])
biases = np.array([2, 3, 0.5, -1, 4])

output = np.dot(inputs, weights.T) + biases
relu_output = np.maximum(0, output)

print(relu_output)
print(relu_output.shape)

print (weights.shape)
print (inputs.shape)
print (biases.shape)
print (weights.T.shape)

