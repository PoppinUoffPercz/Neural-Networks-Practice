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
        self.inputs = inputs  # Store inputs for backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
layer1 = Dense(3, 5)
activation1 = ReLU()
layer2 = Dense(5, 3)

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
loss_function = CrossEntropyLoss()

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

optimizer = SGD(learning_rate=0.01)

for epoch in range(5000):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    softmax.forward(layer2.output)

    loss = loss_function.forward(softmax.output, y_true)

    predictions = np.argmax(softmax.output, axis=1)
    accuracy = np.mean(predictions == y_true)

    loss_function.backward(softmax.output, y_true)
    layer2.backward(loss_function.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.update_parameters(layer1)
    optimizer.update_parameters(layer2)

    if epoch % 500 == 0:
        print("Epoch:", epoch, "Loss:", loss, "Accuracy:", accuracy)
        print("Predictions:", predictions)
        print("True Labels:", y_true)