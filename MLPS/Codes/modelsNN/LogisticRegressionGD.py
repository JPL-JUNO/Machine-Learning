"""
@Description: LogisticRegressionGD
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-04 15:24:42
"""
import numpy as np


class LogisticRegressionGD:
    def __init__(self, eta: float = .01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=.1, scale=.1, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(np.clip(output, 1e-5, 1)))
                    - ((1 - y).dot(np.log(np.clip((1 - output), 1e-5, 1))))
                    / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -300, 300)))

    def predict(self, X, threshold: float = .5):
        return np.where(self.activation(self.net_input(X)) >= threshold, 1, 0)


if __name__ == '__main__':
    pass
