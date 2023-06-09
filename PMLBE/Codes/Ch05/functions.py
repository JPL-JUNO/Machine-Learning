"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 22:55:47
"""

import numpy as np
from numpy import ndarray


def sigmoid(input: ndarray) -> ndarray:
    return 1.0 / (1 + np.exp(-input))


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


def compute_cost(X: ndarray, y: ndarray, weights: ndarray) -> float:
    predictions = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(predictions) -
                   (1 - y) * np.log(1 - predictions))
    return cost


def train_logistic_regression(X_train: ndarray, y_train: ndarray,
                              max_iter: int = 1000,
                              learning_rate: float = .1,
                              fit_intercept: bool = False) -> ndarray:
    """train a logistic regression model

    Args:
        X_train (ndarray): training data(feature)
        y_train (ndarray): training data(target)
        max_iter (int): number of iterations
        learning_rate (float): rate used in iterate weights
        fit_intercept (bool, optional): with an intercept w_0 or not. Defaults to False.

    Returns:
        ndarray: learned weights
    """
    if fit_intercept:
        intercept = np.ones(shape=(X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        if iteration % 1000 == 0:
            print('Current cost is {0:.4f}'.format(
                compute_cost(X_train, y_train, weights)))
    return weights


def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones(shape=(X.shape[0], 1))
        X = np.hstack((intercept, X))
    assert X.shape[1] == weights.shape[0]
    return compute_prediction(X, weights)


if __name__ == '__main__':
    pass
