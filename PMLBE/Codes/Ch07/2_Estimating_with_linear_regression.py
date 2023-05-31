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


def train_linear_regression(X_train, y_train, learning_date: float,
                            max_iter: int = 100, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones(shape=(X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_date)
        if iteration % 100 == 0:
            print('Loss: {0:.3f} at iteration {1}.'.format(
                compute_cost(X_train, y_train, weights), iteration))
    return weights


def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones(shape=(X.shape[0], 1))
        X = np.hstack((intercept, X))
    assert X.shape[1] == weights.shape[0]
    return compute_predictions(X, weights)


X_train = np.array([[6], [2], [3], [4], [1],
                    [5], [2], [6], [4], [7]])
y_train = np.array([5.5, 1.6, 2.2, 3.7, 0.8,
                    5.2, 1.5, 5.3, 4.4, 6.8])
weights = train_linear_regression(
    X_train, y_train, learning_date=.01, fit_intercept=True)
X_test = np.array([[1.3], [3.5], [5.2], [2.8]])
predictions = predict(X_test, weights)

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, marker='o', c='b')
plt.scatter(X_test[:, 0], predictions, marker='*', c='k')
plt.xlabel('x')
plt.ylabel('y')

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
assert diabetes.data.shape == (442, 10)

num_test = 30
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
weights = train_linear_regression(
    X_train, y_train, max_iter=5000, learning_date=1, fit_intercept=True)
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]
predictions = predict(X_test, weights)


# Implementing linear regression with scikit-learn
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(loss='squared_error', penalty='l2',
                         alpha=.0001, learning_rate='constant', eta0=.01, max_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


import tensorflow as tf
layer0 = tf.keras.layers.Dense(units=1, input_shape=[X_train.shape[1]])
model = tf.keras.Sequential(layer0)

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1))
model.fit(X_train, y_train, epochs=100, verbose=True)
predictions = model.predict(X_test)[:, 0]
