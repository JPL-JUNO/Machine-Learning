"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-01 22:12:13
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def train(X, y, n_hidden, learning_rate, n_iter: int):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros(shape=(1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros(shape=(1, 1))
    for i in range(1, n_iter + 1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)

        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3
        dZ3 = Z3 - y
        dW2 = np.matmul(A2.T, dZ3)
