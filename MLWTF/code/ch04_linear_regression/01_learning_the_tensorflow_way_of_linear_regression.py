"""
@Title: Learning the TensorFlow way of linear regression
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-10 21:44:31
@Description: 
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from pandas import DataFrame
from typing import Iterable, Callable

housing_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
path = tf.keras.utils.get_file(housing_url.split("/")[-1],
                               housing_url)
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_table(path, delim_whitespace=True,
                     header=None, names=columns)

np.random.seed(1)
train = data.sample(frac=.8).copy()
y_train = train["MEDV"]
train = train.drop("MEDV", axis=1)
# loc 基于显式索引
test = data.loc[~data.index.isin(train.index)].copy()
y_test = test["MEDV"]
test = test.drop("MEDV", axis=1)

learning_rate = .05


def make_input_fn(data_df, label_df, num_epochs=10,
                  shuffle=True, batch_size=256) -> Callable:
    """
    creates a tf.data dataset from a pandas 
    DataFrame turned into a Python dictionary of pandas Series (the features are the
    keys, the values are the feature vectors).
    这个应该是提供一个函数映射，用于两个函数的实例化
    """
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


def define_feature_columns(data_df: DataFrame, categorical_cols: Iterable,
                           numeric_cols: Iterable) -> list:
    """
    maps each column name to a specific `tf.feature_column` transformation.
    有弃用警告
    """
    feature_columns = []
    for feature_name in numeric_cols:
        # 弃用
        feature_columns.append(tf.feature_column.numeric_column(
            feature_name, dtype=tf.float32
        ))
    for feature_name in categorical_cols:
        vocabulary = data_df[feature_name].unique()
        # tf.keras.layers.CategoryEncoding()
        feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name, vocabulary
            ))
    return feature_columns


categorical_cols = ['CHAS', 'RAD']
numeric_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM',
                'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
feature_columns = define_feature_columns(data, categorical_cols,
                                         numeric_cols)
train_input_fn = make_input_fn(train, y_train, num_epochs=1_400)
test_input_fn = make_input_fn(test, y_test, num_epochs=1, shuffle=False)

# 弃用
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(test_input_fn)
