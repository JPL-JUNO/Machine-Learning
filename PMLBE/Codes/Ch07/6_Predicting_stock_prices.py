"""
@Description: Predicting stock prices with the three regression algorithms
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-01 11:01:32
"""

import pandas as pd
from data_generate import generate_features, change_str_to_float
data_raw = pd.read_csv('../data/19880101_20191231.csv',
                       index_col='Date', parse_dates=['Date'])
change_str_to_float(data_raw)
data = generate_features(data_raw)


start_train = '1988-01-04'
end_train = '2017-12-31'
start_test = '2018-01-04'
end_test = '2018-12-27'

data_train = data.loc[start_train:end_train]
data_test = data.loc[start_test:end_test]
X_train = data_train.drop('Close', axis=1).values
y_train = data_train['Close'].values
X_test = data_test.drop('Close', axis=1).values
y_test = data_test['Close'].values
assert X_train.shape == (7561, 37)
assert y_train.shape == (7561,)
assert X_test.shape == (247, 37)
assert y_test.shape == (247,)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
param_grid = {'alpha': [1e-4, 3e-4, 1e-3],
              'eta0': [.01, .03, .1]}
lr = SGDRegressor(penalty='l2', max_iter=1000, random_state=42)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
grid_search.fit(X_scaled_train, y_train)
lr_best = grid_search.best_estimator_
predictions_lr = lr_best.predict(X_scaled_test)


param_grid = {'max_depth': [30, 50], 'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [3, 5]}
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,
                           random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)

param_grid = [
    {'kernel': ['linear'], 'C':[100, 200, 300], 'epsilon':[.00003, .0001]},
    {'kernel': ['rbf'], 'gamma':[1e-3, 1e-4],
        'C':[10, 100, 1000], 'epsilon': [.00003, .0001]}
]
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_scaled_train, y_train)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)

import matplotlib.pyplot as plt
plt.plot(data_test.index, y_test, c='k', label='Truth')
plt.plot(data_test.index, predictions_lr, c='b', label='LR')
plt.plot(data_test.index, predictions_rf,
         c='r', label='RFR')
plt.plot(data_test.index, predictions_svr, c='r', label='SVR')
