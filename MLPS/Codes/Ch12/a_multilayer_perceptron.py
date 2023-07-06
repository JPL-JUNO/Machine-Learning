"""
@Description: Building a multilayer perceptron for classifying flowers in the Iris dataset
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-06 11:40:35
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1. / 3, random_state=1)
import torch
import numpy as np
X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
