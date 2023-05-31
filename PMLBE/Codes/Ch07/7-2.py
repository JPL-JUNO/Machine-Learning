"""
@Description: Implementing linear regression from scratch
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-31 16:29:52
"""
import numpy as np


def compute_predictions(X, weights):
    predictions = np.dot(X, weights)
    return predictions


def update_weights_gd(X_train, y_train,
                      weights, learning_date):
    predictions = compute_predictions(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_date / float(m) * weights_delta
    return weights


def compute_cost(X, y, weights):
    predictions = compute_predictions(X, weights)
    cost = np.mean((predictions - y)**2 / 2.0)
    return cost


def train_linear_regression(X_train, y_train, max_iter,
                            learning_date, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones(shape=(X_train.shape[1], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_date)
        if iteration % 100 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights
