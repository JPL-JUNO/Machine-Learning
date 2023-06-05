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
    def __init__(self, path: str = './'):
        self.path = path

    def get_iris_data(self, filename: str = 'iris', extension='.csv',
                      train_test_split: bool = True):
        url = self.path + filename + extension
        df = pd.read_csv(url, header=None, low_memory=True)
        X = df.iloc[:, :-1].values
        y = df[-1].values

        if train_test_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y)
