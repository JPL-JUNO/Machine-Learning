"""
@Description: get date to use
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-05 22:14:04
"""

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None

    def get_iris(self):
        self.data = pd.read_csv('../data/iris.data', header=None)
        return self.data

    def get_wine(self):
        self.data = pd.read_csv('../data/wine.data', header=None)
        self.X = self.data.iloc[:, 1:].values
        self.y = self.data.iloc[:, 0].values
        return self.data

    def standardize(self):
        pass

    def get_train_test(self, test_size: float = .2, random_state: int = 42):
        assert test_size < 1, '不建议指定数量，需指定测试样本比例'
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
