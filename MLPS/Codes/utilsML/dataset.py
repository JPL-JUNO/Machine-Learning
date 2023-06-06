"""
@Description: functions used to get datasets
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-03 21:52:36
"""

import pandas as pd
import numpy as np
from numpy import ndarray


def get_iris() -> tuple[ndarray, ndarray]:
    df = pd.read_csv('../data/iris.data', header=None, encoding='utf-8')

    y = df.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[:, [0, 2]].values
    return X, y


if __name__ == '__main__':
    get_iris()
