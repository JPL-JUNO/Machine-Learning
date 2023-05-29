"""
@Description: Ensembling decision trees–gradient boosted trees
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 16:05:30
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

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
import xgboost as xgb
model = xgb.XGBClassifier(learning_rate=.1, max_depth=10, n_estimators=1000)

model.fit(X_train_enc, Y_train_enc)
pos_prob = model.predict_proba(X_test_enc)[:, 1]
from sklearn.metrics import roc_auc_score
print('The ROC AUC on testing set is {0:.3f}'.format(
    roc_auc_score(Y_test, pos_prob)))
