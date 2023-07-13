"""
@Description: Choosing activation functions for multilayer neural networks
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-10 14:05:29
"""

import numpy as np
from torch import Tensor
import torch

X = np.array([1, 1.4, 2.5])
w = np.array([.4, .3, .5])


def net_input(X: Tensor, w: Tensor) -> Tensor:
    return np.dot(X, w)


def logistic(z: Tensor) -> Tensor:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_activation(X: Tensor, w: Tensor) -> Tensor:
    z = net_input(X, w)
    return logistic(z)


print(f'P(y=1|x) = {logistic_activation(X, w):.3f}')

W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])
A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('Net Input: \n', Z)
print('Output Units:\n', y_probas)

y_class = np.argmax(Z, axis=0)
print("Predicted class label:", y_class)

# Estimating class probabilities in multiclass classification
# via the softmax function


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
print(np.sum(y_probas))

print(torch.softmax(torch.from_numpy(Z), dim=0))

import matplotlib.pyplot as plt


def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5, .005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.plot(z, tanh_act, label="tanh", linewidth=3, linestyle='--')
plt.plot(z, log_act, linewidth=3, label='logistic')
plt.legend()
plt.xlabel('net input $z$')
plt.ylabel('activation $\phi(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')
plt.ylim([-1.5, 1.5])
plt.tight_layout()
plt.show()


print(np.tanh(z))
print(torch.tanh(torch.from_numpy(z)))

from scipy.special import expit
print(expit(z))
print(torch.sigmoid(torch.from_numpy(z)))
# Note that using torch.sigmoid(x) produces results that are equivalent to torch.
# nn.Sigmoid()(x), which we used earlier. torch.nn.Sigmoid is a class to which you
# can pass in parameters to construct an object in order to control the behavior. In contrast,
# torch.sigmoid is a function.

print(torch.relu(torch.from_numpy(z)))
