"""
@Description: models implemented to used in this chapter
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-03 22:31:40
"""
import numpy as np
from numpy import ndarray


class AdalineSGD:
    def __init__(self, eta: float = .01, n_iter: int = 10,
                 shuffle: bool = True, random_state: int = 42) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: ndarray, y: ndarray):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, label in zip(X, y):
                losses.append(self._update_weights(xi, label))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= .5, 0, 1)

    def _shuffle(self, X, y):
        # pass test that the result differ each time though the random seed
        r = self.rng.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rng = np.random.RandomState(self.random_state)
        self.w_ = self.rng.normal(loc=.0, scale=.1, size=m)
        self.b_ = np.float_(.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * error * xi
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss


if __name__ == "__main__":
    pass
