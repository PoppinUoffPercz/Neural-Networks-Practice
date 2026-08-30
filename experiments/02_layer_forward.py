import numpy as np

inputs = np.array([
    [1, 2, 3],
    [3, 2, 1],
    [0, 1, 2],
    [4, 3, 2],
    [-1, 0, 1]
])

y_true = np.array([0, 1, 2, 1, 0])  # Example true labels for the inputs

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs  # Store inputs for backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
layer1 = Dense(3, 5)
activation1 = ReLU()

layer1.forward(inputs)
activation1.forward(layer1.output)

print(activation1.output)
print(activation1.output.shape)
print(layer1.output.shape)

layer2 = Dense(5, 3)

layer2.forward(activation1.output)


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred_clipped[
            range(samples),
            y_true
        ]

        losses = -np.log(correct_confidences)

        return np.mean(losses)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        self.dinputs = y_pred.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples

softmax = Softmax()
softmax.forward(layer2.output)
loss_function = CrossEntropyLoss()

loss = loss_function.forward(
    softmax.output,
    y_true
)
print(softmax.output.shape)
print(softmax.output)
print(np.sum(softmax.output, axis=1))  # Should be close to 1 for each sample

loss_function.backward(softmax.output, y_true)
layer2.backward(loss_function.dinputs)
print(loss_function.dinputs)
print(loss_function.dinputs.shape)