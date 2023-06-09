"""
@Description: Predicting ad click-through with logistic regression using gradient descent
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 11:22:02
"""

import pandas as pd
n_rows = 300_000
df = pd.read_csv('../data/train.csv', nrows=n_rows)
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = 10_000
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


import timeit
from functions import train_logistic_regression
from functions import predict
from sklearn.metrics import roc_auc_score

start_time = timeit.default_timer()
weights = train_logistic_regression(X_train_enc.toarray(),
                                    Y_train, max_iter=10_000,
                                    learning_rate=.01, fit_intercept=True)
end_time = timeit.default_timer()
print('Running time(s): {0:.3f}s'.format(end_time - start_time))
pred = predict(X_test_enc.toarray(), weights)
print('Training samples: {0}, AUC on testing set: {1:.3f}'.format(n_train,
                                                                  roc_auc_score(Y_test, pred)))
