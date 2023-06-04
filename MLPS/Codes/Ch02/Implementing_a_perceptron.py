"""
@Description: Implementing a perceptron learning algorithm in Python
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-02 14:01:02
"""

import numpy as np
from numpy import ndarray
import sys
sys.path.append('../utils')
from visualize import plot_decision_regions


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

if __name__ == '__main__':
    import os
    import pandas as pd
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    # df = pd.read_csv(url, header=None, encoding='utf-8')
    df = pd.read_csv('../data/iris.data', header=None, encoding='utf-8')

    import matplotlib.pyplot as plt
    import numpy as np

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[0:100, [0,2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()
    
    ppn = Perceptron(eta=.1, n_iter=10)
    ppn.fit(X, y)
    fig = plt.figure()
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    
    fig = plt.figure()
    plot_decision_regions(X, y, ppn)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()