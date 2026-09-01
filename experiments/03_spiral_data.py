import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

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
class SGD:
    def __init__(self, learning_rate=0.01, decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = (
                self.learning_rate /
                (1 + self.decay * epoch)
            )

    def update_parameters(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases -= self.current_learning_rate * layer.dbiases

samples = 100
classes = 3

X = np.zeros((samples * classes, 2))
y = np.zeros(samples * classes, dtype='uint8')

for class_number in range(classes):
    ix = range(
        samples * class_number,
        samples * (class_number + 1)
    )
    r = np.linspace(0.0, 1, samples)
    t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
    X[ix] = np.c_[r * np.sin(t*2.5), r * np.cos(t*2.5)]
    y[ix] = class_number

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.title("Spiral Dataset")
plt.show()

layer1 = Dense(2, 64)
activation1 = ReLU()
layer2 = Dense(64, 3)
softmax = Softmax()
loss_function = CrossEntropyLoss()
optimizer = SGD(
    learning_rate=0.52,
    decay=0.0005
)

loss_history = []

for epoch in range(5000):
    optimizer.pre_update_params()
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    softmax.forward(layer2.output)

    loss = loss_function.forward(softmax.output, y)
    loss_history.append(loss)

    predictions = np.argmax(softmax.output, axis=1)
    accuracy = np.mean(predictions == y)

    loss_function.backward(softmax.output, y)
    layer2.backward(loss_function.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()

    optimizer.update_parameters(layer1)
    optimizer.update_parameters(layer2)

    if epoch % 500 == 0:
        print(
            "Epoch:", epoch, 
            "Loss:", loss,
            "Accuracy:", accuracy,
            "LR:", optimizer.current_learning_rate
        )

x_min = X[:, 0].min() - 0.1
x_max = X[:, 0].max() + 0.1
y_min = X[:, 1].min() - 0.1
y_max = X[:, 1].max() + 0.1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
layer1.forward(grid_points)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
softmax.forward(layer2.output)
grid_predictions = np.argmax(softmax.output, axis=1)
grid_predictions = grid_predictions.reshape(xx.shape)

plt.figure()

plt.contourf(
    xx,
    yy,
    grid_predictions,
    alpha=0.3,
    cmap="brg"
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="brg"
)

plt.title("Neural Network Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()