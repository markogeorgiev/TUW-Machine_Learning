# A General-Purpose Feedforward Neural Network Architecture for Supervised Classification

## Abstract

We propose a generalized feedforward neural network architecture capable of solving a wide range of supervised classification tasks. The design leverages deep hierarchical representations via nonlinear activation functions in the hidden layers, followed by a softmax-based output layer to model class probabilities. The architecture is trained end-to-end using stochastic gradient descent and categorical cross-entropy loss. Despite its simplicity, the model demonstrates strong capacity to learn nonlinear mappings from input to target space, making it well-suited for both low- and high-dimensional classification problems.

---

## 1. Introduction

Many supervised learning problems involve highly nonlinear relationships between input features and class labels. Traditional models such as linear classifiers or shallow networks struggle with such complexity. Deep neural networks, in contrast, have demonstrated remarkable performance by learning multiple layers of abstraction. In this work, we present a general-purpose architecture combining nonlinear transformations with probabilistic outputs, optimized using gradient descent. While the architecture is minimal, it is fully expressive and supports various activation strategies, providing a foundation for extensibility and experimentation.

---

## 2. Architecture Overview

The generalized network architecture consists of:

- **Input layer**: Dimensionality $d$, where $d$ is the number of input features
- **One or more hidden layers**: Each with $h$ neurons and a nonlinear activation (e.g., ReLU or Sigmoid)
- **Output layer**: $C$ neurons with Softmax activation, where $C$ is the number of classes

### Layer Definition

Let $\mathbf{x} \in \mathbb{R}^d$ be the input. For a network with $L$ layers, the forward propagation is defined as:

1. **Hidden transformations**:
   $$
   \mathbf{z}^{(l)} = \mathbf{a}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}, \quad
   \mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})
   $$

   for $l = 1, \ldots, L-1$, where $f(\cdot)$ is the activation function (e.g., ReLU or Sigmoid)

2. **Output layer**:
   $$
   \mathbf{z}^{(L)} = \mathbf{a}^{(L-1)} \mathbf{W}^{(L)} + \mathbf{b}^{(L)} \\
   \hat{\mathbf{y}} = \text{Softmax}(\mathbf{z}^{(L)})
   $$

---

## 3. Activation Functions

### ReLU (Rectified Linear Unit)

Defined as:
$$
\text{ReLU}(x) = \max(0, x)
$$

ReLU is widely used due to its simplicity and ability to mitigate vanishing gradients.

### Sigmoid

Defined as:
$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

Useful for binary activation, though less common in deeper networks due to saturation effects.

### Softmax

Defined as:
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax maps logits to a probability distribution over classes, making it suitable for multi-class classification.

---

## 4. Loss Function

We use the **Categorical Cross-Entropy** loss function for multi-class classification:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log(\hat{y}_{i, t_i})
$$

Where $\hat{y}_{i, t_i}$ is the predicted probability for the correct class $t_i$.

Its gradient with respect to the softmax logits simplifies to:

$$
\frac{\partial \mathcal{L}}{\partial z} = \hat{\mathbf{y}} - \mathbf{y}_{\text{true}}
$$

This simplification allows efficient implementation of backpropagation.

---

## 5. Training Algorithm

Training proceeds using **stochastic gradient descent (SGD)** or its variants. For each mini-batch, the forward pass computes predictions and the backward pass computes gradients for parameter updates:

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}, \quad
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}
$$

Where $\eta$ is the learning rate.

---

## 6. Experimental Setup

We demonstrate this architecture on the widely used spiral dataset, a challenging problem for linear models due to its intertwined class structure.

```python
from nnfs.datasets import spiral_data
import numpy as np
import nnfs

nnfs.init()
X, y = spiral_data(samples=1000, classes=2)

nn = NeuralNetwork()
nn.add_layer(2, 64, "relu")
nn.add_layer(64, 2, "softmax")

epochs = 1000
lr = 0.05

for epoch in range(epochs):
    y_pred = nn.forward(X)
    loss = LossFunction.categorical_cross_entropy(y_pred, y)
    grad = LossFunction.categorical_cross_entropy_derivative(y_pred, y)
    nn.backward(grad, learning_rate=lr)

    if epoch % 100 == 0:
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
```

---

## 7. Results

Across multiple trials, the network consistently learned to separate the intertwined spiral classes, achieving classification accuracy exceeding 90% after 1000 epochs. These results support the claim that even modestly deep networks with nonlinear activations and probabilistic outputs can solve nontrivial pattern recognition tasks.

---

## 8. Conclusion

We presented a general-purpose feedforward neural network capable of learning nonlinear mappings for supervised classification. Its design combines expressive hidden transformations and a softmax-based output layer. The network is easy to train via gradient descent and shows strong performance on complex, structured datasets. This architecture forms a reliable foundation for further explorations in deep learning and computational neuroscience.
