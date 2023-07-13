"""
@Description: Linear Regression Gradient Descent
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-23 20:52:00
"""
import numpy as np


class LinearRegressionGD:
    def __init__(self, eta: float = .01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_ += 2.0 * self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += 2.0 * self.eta * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)
