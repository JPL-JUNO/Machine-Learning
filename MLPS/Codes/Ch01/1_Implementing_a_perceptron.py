"""
@Description: Implementing a perceptron learning algorithm in Python
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-02 14:01:02
"""

import numpy as np
from numpy import ndarray


class Perceptron:
    def __init__(self, eta: float = .001,
                 n_iter: int = 100, random_state: int = 42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: ndarray, y: ndarray):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0, scale=.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        # 保留每次迭代的错误率
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: ndarray):
        """计算网络的输入

        Args:
            X (ndarray): feature used to train

        Returns:
            _type_: _description_
        """
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: ndarray) -> int:
        """return class label after unit step

        Args:
            X (ndarray): feature

        Returns:
            int: class label
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
