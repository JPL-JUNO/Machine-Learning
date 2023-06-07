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
        self.data = pd.read_csv('wine.data', header=None)
        return self.data

    def standardize(self):
        pass

    def get_train_test(self, test_size: float = .2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size)
        return X_train, X_test, y_train, y_test
