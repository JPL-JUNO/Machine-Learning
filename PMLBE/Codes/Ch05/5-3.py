"""
@Description: Training a logistic regression model using gradient descent
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 22:54:15
"""

import numpy as np
from functions import sigmoid


def compute_prediction(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions


def update_weights_gd(X_train, Y_train, weights, learning_rate: float):
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, Y_train - predictions)
    m = Y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights
