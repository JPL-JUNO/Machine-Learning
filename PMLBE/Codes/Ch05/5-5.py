"""
@Description: Training a logistic regression model using stochastic gradient descent
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 14:13:27
"""
import numpy as np
from numpy import ndarray
from functions import compute_prediction
from functions import compute_cost
from functions import predict


def update_weights_sgd(X_train, Y_train, weights, learning_rate: float):
    for X_each, y_each in zip(X_train, Y_train):
        prediction = compute_prediction(X_each, weights)
        weights_delta = X_each.T * (y_each - prediction)
        weights += learning_rate * weights_delta
    return weights


def train_logistic_regression_sgd(X_train, y_train, max_iter,
                                  learning_rate, fit_intercept: bool = False):
    if fit_intercept:
        intercept = np.ones(shape=(X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_sgd(X_train, y_train, weights, learning_rate)
        if iteration % 2 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights


import pandas as pd
n_rows = 300_000
df = pd.read_csv('../data/train.csv', nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = 100_000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

import timeit
start_time = timeit.default_timer()
weights = train_logistic_regression_sgd(X_train_enc.toarray(), Y_train,
                                        max_iter=10, learning_rate=.01, fit_intercept=True)
end_time = timeit.default_timer()
print('Running time(s): {0:.3f}s'.format(end_time - start_time))
pred = predict(X_test_enc.toarray(), weights)
print('Training samples: {0}, AUC on testing set: {1:.3f}'.format(n_train,
                                                                  roc_auc_score(Y_test, pred)))

from sklearn.linear_model import SGDClassifier
sgd_lr = SGDClassifier(loss='log_loss', penalty=None,
                       fit_intercept=True, max_iter=20,
                       learning_rate='constant', eta0=.01)
# mean the learning rate is 0.01 and unchanged during the course of training
# It should be noted that the default learning_rate is 'optimal', where the
# learning rate slightly decreases as more and more updates are made. This can be
# beneficial for finding the optimal solution on large datasets.
sgd_lr.fit(X_train_enc.toarray(), Y_train)
pred = sgd_lr.predict_proba(X_test_enc.toarray())[:, 1]
print('Training samples:{0}, AUC on testing set: {1:.3f}'.format(
    n_train, roc_auc_score(Y_test, pred)))


sgd_lr_l1 = SGDClassifier(loss='log_loss', penalty='l1', alpha=.0001,
                          fit_intercept=True, max_iter=20,
                          learning_rate='constant', eta0=.01)
sgd_lr_l1.fit(X_train_enc.toarray(), Y_train)
coef_abs = np.abs(sgd_lr_l1.coef_)
feature_names = enc.get_feature_names_out()
bottom_10 = np.argsort(coef_abs)[0][:10]
print('10 least important features are: \n', feature_names[bottom_10])

top_10 = np.argsort(coef_abs)[0][-10:]
print('10 least important features are: \n', feature_names[top_10])
