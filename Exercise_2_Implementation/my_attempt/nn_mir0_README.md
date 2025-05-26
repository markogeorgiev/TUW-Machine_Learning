# Neural Network From Scratch - Documentation

This document provides a comprehensive guide for using a NumPy-based neural network implementation for classification tasks, including model architecture, training, loss functions, and evaluation.

---

## Components

### NeuralNetwork
- Manages a stack of layers.
- Performs forward and backward propagation.
- Updates parameters via gradient descent.

### Layer(input_dim, output_dim, activation)
- Defines a layer with weights `W`, biases `b`, and activation function.
- Activation functions supported: `relu`, `sigmoid`, `softmax`, `linear`.
- Stores intermediate values for backpropagation.

### ActivationFunction
- Provides activation functions and their derivatives:
  - `relu`, `relu_derivative`
  - `sigmoid`, `sigmoid_derivative`
  - `linear`, `linear_derivative`
  - `softmax` (no derivative used directly)

### LossFunction
- Includes loss functions for training:
  - `mse`, `mse_derivative`
  - `categorical_cross_entropy`, `categorical_cross_entropy_derivative`

---

## Usage Guide

### 0. Imports
Import the nn_mir0.py file 

### 1. Define Model Architecture

```python
nn = NeuralNetwork()
nn.add_layer(input_dim=36, output_dim=6, activation="relu")
nn.add_layer(input_dim=6, output_dim=3, activation="softmax")
```

You can define model variants programmatically:

```python
def build_nn(model_type, input_dim, output_dim):
    nn = NeuralNetwork()
    if model_type == "1_relu":
        nn.add_layer(input_dim, 6, "relu")
        nn.add_layer(6, output_dim, "softmax")
    # ... other configurations
    return nn
```

### 2. Forward Pass

```python
y_pred = nn.forward(X_train)
```

### 3. Compute Loss

```python
loss = LossFunction.categorical_cross_entropy(y_pred, y_train)  # y_train must be class indices
```

### 4. Backpropagation and Weight Update

```python
loss_grad = LossFunction.categorical_cross_entropy_derivative(y_pred, y_train)
nn.backward(loss_grad, learning_rate=0.05)
```

### 5. Prediction and Evaluation

```python
predictions = np.argmax(y_pred, axis=1)
accuracy = np.mean(predictions == y_train)
```

---

## Training Loop Example

```python
for epoch in range(epochs):
    y_pred = nn.forward(X_train)
    loss = LossFunction.categorical_cross_entropy(y_pred, y_train)
    grad = LossFunction.categorical_cross_entropy_derivative(y_pred, y_train)
    nn.backward(grad, learning_rate=lr)

    if epoch % 10 == 0 or epoch == epochs - 1:
        acc = np.mean(np.argmax(y_pred, axis=1) == y_train)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
```

---

## Best Practices

- Use `softmax` activation in the output layer for multi-class classification.
- `categorical_cross_entropy` expects `targets` as integer class indices.
- Learning rate and number of epochs should be tuned based on performance.
- Always normalize or scale input features when working with real-world datasets.

---

## Example Implementation

```python
from nnfs.datasets import spiral_data
import numpy as np
import nnfs

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
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
```

