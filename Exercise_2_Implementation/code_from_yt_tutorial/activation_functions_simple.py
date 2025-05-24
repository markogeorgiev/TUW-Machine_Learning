import numpy as np
import nnfs

nnfs.init()

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
# print(exp_values)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)





# ===============================================
'''for output in layer_outputs:
     exp_values.append(E**output)'''

'''#print(exp_values)

norm_base = np.sum(exp_values)
norm_values = []

for value in exp_values:
     norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values))'''
