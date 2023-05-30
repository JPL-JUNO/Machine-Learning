"""
@Description: Implementing logistic regression using TensorFlow
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 16:46:39
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
n_rows = 300_000
df = pd.read_csv('../data/train.csv', nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
n_train = 100_000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

X_train_enc = X_train_enc.toarray().astype('float32')
X_test_enc = X_test_enc.toarray().astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

batch_size = 1_000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

n_features = int(X_train_enc.shape[1])
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))
