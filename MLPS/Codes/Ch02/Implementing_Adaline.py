"""
@Description: Implementing Adaline in Python
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-03 18:18:26
"""
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('..')
from utils.dataset import get_iris
from utils.visualize import plot_decision_regions

class AdalineGD:
    def __init__(self, eta: float = .01, n_iter: int = 50,
                 random_state: int = 42) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=.1, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            # for w_j in range(self.w_.shape[0]):
            #     self.w_[w_j] += 2.0 * self.eta * (errors * X[:, w_j]).mean()
            # 下面这一行只是使用了矩阵的形式来进行计算，避免了使用loop
            self.w_ += self.eta * 2 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X: ndarray) -> ndarray:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: ndarray) -> ndarray:
        return X

    def predict(self, X: ndarray):
        return np.where(self.activation(self.net_input(X)) >= .5, 0, 1)

X, y = get_iris()
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
ada1 = AdalineGD(n_iter=15, eta=.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_)+1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log10(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = AdalineGD(n_iter=15, eta=.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_)+1), ada2.losses_, marker='s')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std = (X_std - X_std.mean(axis=0))/X_std.std(axis=0)

ada_gd = AdalineGD(n_iter=20, eta=.5)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_gd.losses_) + 1),
ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()