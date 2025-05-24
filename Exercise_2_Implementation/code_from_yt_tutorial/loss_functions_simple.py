import math

import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

'''loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print(math.log(1))
print(math.log(0.9))
print(math.log(0.8))
print(math.log(0.7))
print(math.log(0.6))
print(math.log(0.5))
print(math.log(0.4))
print(math.log(0.3))
print(math.log(0.2))
print(math.log(0.1))
print(math.log(0.1))
print(math.log(0.01))
print(math.log(0.001))
print(math.log(0.0001))'''
