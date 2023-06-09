"""
@Description: Training on large datasets with online learning
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 16:09:11
"""

n_rows = 100_000 * 11
import pandas as pd
df = pd.read_csv('../data/train.csv', nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values
n_train = 100000 * 10
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import timeit
batch_size = 100_000
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
sgd_lr_online = SGDClassifier(loss='log_loss', penalty=None,
                              fit_intercept=True, max_iter=1,
                              learning_rate='constant', eta0=.01)
start_time = timeit.default_timer()
for i in range(10):
    x_train = X_train[i * batch_size: (i + 1) * batch_size]
    y_train = Y_train[i * batch_size: (i + 1) * batch_size]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc.toarray(), y_train, classes=[0, 1])
print('Online learning time(s): {0:.3f}s'.format(
    timeit.default_timer() - start_time))
x_test_enc = enc.transform(X_test)
pred = sgd_lr_online.predict_proba(x_test_enc.toarray())[:, 1]
print('Training samples: {0}, AUC on testing set: {1:.3f}'.format(
    n_train, roc_auc_score(Y_test, pred)))
