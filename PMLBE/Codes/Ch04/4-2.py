"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-28 22:00:19
"""

import numpy as np
from numpy import ndarray
from typing import Tuple


def gini_impurity_np(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_count=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


def entropy_np(labels):
    if labels.size == 0:
        return 0
    counts = np.unique(labels, return_count=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions * np.log2(fractions))


criterion_function = {'gini': gini_impurity_np,
                      'entropy': entropy_np}


def weighted_impurity(groups, criterions: str = 'gini'):
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * \
            criterion_function[criterions](group)
    return weighted_sum


def split_node(X: ndarray, y: ndarray, index: int, value) -> Tuple[list, list]:
    """split dataset X, y based on a feature and value

    Args:
        X (ndarray): dataset feature
        y (ndarray): dataset target
        index (int): index of the feature used for splitting
        value (_type_): value of the feature used for splitting

    Returns:
        Tuple[ndarray, ndarray]: a child is in the format of [X, y]
    """
    x_index = X[:, index]
    # if this feature is numerical
    if X[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    # if this feature is categorical
    else:
        mask = x_index == value
    # split into left and right
    left = [X[~mask, :], y[~mask]]
    right = [X[mask, :], y[mask]]
    return left, right


def get_best_split(X: ndarray, y: ndarray, criterion: str = 'gini'):
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(X[0])):
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity(
                [groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}
