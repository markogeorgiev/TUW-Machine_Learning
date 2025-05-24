import numpy as np

np.random.seed(0)

class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        pass


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.rand(input_dim, output_dim)
        self.b = np.random.rand(1, output_dim)
        self.activation_name = activation

        self.activation = getattr(ActivationFunction, activation)
        self.activation_derivative = getattr(ActivationFunction, f"{activation}_derivative")

        self.z = None
        self.a = None
        self.input = None

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output):
        if self.activation_name not in ["linear", "softmax"]:
            dz = grad_output * self.activation_derivative(self.z)
        else:
            dz = grad_output

        dW = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        grad_input = np.dot(dz, self.W.T)
        return grad_input, dW, db


class LossFunction:
    @staticmethod
    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mse_derivative(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        pass

    @staticmethod
    def categorical_cross_entropy(predictions, targets):
        samples = predictions.shape[0]
        clipped_preds = np.clip(predictions, 1e-7, 1 - 1e-7)
        correct_confidences = clipped_preds[range(samples), targets]
        loss = -np.log(correct_confidences)
        return np.mean(loss)

    @staticmethod
    def categorical_cross_entropy_derivative(predictions, targets):
        samples = predictions.shape[0]
        grad = predictions.copy()
        grad[range(samples), targets] -= 1
        return grad / samples


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation):
        layer = Layer(input_dim, output_dim, activation)
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads, dW, db = layer.backward(grads)
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db


'''from nnfs.datasets import vertical_data, spiral_data
import numpy as np

nnfs.init()

X, y = spiral_data(samples=1000, classes=2)

# Initialize network
nn = NeuralNetwork()
nn.add_layer(2, 64, "relu")
nn.add_layer(64, 2, "softmax")

epochs = 1000
lr = 0.05

for epoch in range(epochs):
    y_pred = nn.forward(X)
    loss = LossFunction.categorical_cross_entropy(y_pred, y)
    loss_grad = LossFunction.categorical_cross_entropy_derivative(y_pred, y)
    nn.backward(loss_grad, learning_rate=lr)

    if epoch % 100 == 0:
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")'''