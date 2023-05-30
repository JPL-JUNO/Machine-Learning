"""
@Description: Feature selection using random forest
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 21:58:06
"""
import pandas as pd
import numpy as np
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
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,
                                       criterion='gini', min_samples_split=30, n_jobs=-1)
random_forest.fit(X_train_enc.toarray(), Y_train)
feature_imp = random_forest.feature_importances_
feature_names = enc.get_feature_names_out()
bottom_10 = np.argsort(feature_imp)[:10]
print('10 least important features are: \n{0}'.format(
    feature_names[bottom_10]))
