"""
@Description: predicting ad click-through with a decision tree
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 11:15:46
"""


import pandas as pd

n_rows = 300_000
df = pd.read_csv('train.csv', nrows=n_rows)
Y = df['click'].values
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
assert X.shape == (n_rows, 19)

n_train = int(n_rows * .9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [3, 10, None]}
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters,
                           n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]
from sklearn.metrics import roc_auc_score
print('The ROC AUC on testing set is: {0:.3f}'.format(
    roc_auc_score(Y_test, pos_prob)))

# random predict
import numpy as np
pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test), int(
    len(Y_test) * 51211 / n_rows), replace=False)
pos_prob[click_index] = 1
print('The ROC AUC on testing set is: {0:.3f}'.format(
    roc_auc_score(Y_test, pos_prob)))
