import sys
import numpy as np
import matplotlib.pyplot as plt

from Exercise_2_Implementation.neuron_and_layer_simple import outputs

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

outputs = np.dot(weights, inputs) + biases


print(outputs)

