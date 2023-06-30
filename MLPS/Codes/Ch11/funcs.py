"""
@Description: functions that provide convenience
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-30 16:46:51
"""

import numpy as np
from numpy import ndarray


def sigmoid(z: ndarray) -> ndarray:
    """
    _summary_

    Parameters
    ----------
    z : ndarray
        _description_

    Returns
    -------
    ndarray
        _description_
    """
    return 1 / (1 + np.exp(-z))


def int_to_onehot(y: ndarray, num_labels: int) -> ndarray:
    """
    将一个预测向量转化为ndarray形式的独热编码

    Parameters
    ----------
    y : ndarray
        预测向量
    num_labels : int
        不同类型标签的数量

    Returns
    -------
    ndarray
        独热形式表示的预测向量
    Examples:
    >>> int_to_onehot(np.array([1, 3, 5, 6]), 7)
    >>> array([[0., 1., 0., 0., 0., 0., 0.],
    ...        [0., 0., 0., 1., 0., 0., 0.],
    ...        [0., 0., 0., 0., 0., 1., 0.],
    ...        [0., 0., 0., 0., 0., 0., 1.]])

    """
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary
