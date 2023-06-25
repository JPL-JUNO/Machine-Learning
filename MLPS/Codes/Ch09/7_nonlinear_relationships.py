"""
@Description: Dealing with nonlinear relationships using random forests
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-25 22:07:15
"""
from sklearn.tree import DecisionTreeRegressor
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('./')
sys.path.append('../')
from utilsML.visualize import lin_reg_plot
from utilsML.visualize import plot_residual

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('AmesHousing.txt', sep='\t', usecols=columns)
df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})
df = df.dropna(axis=0)
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()

fig, ax = plt.subplots()
lin_reg_plot(X[sort_idx], y[sort_idx], tree, ax)
plt.xlabel('Living area above ground in square feet')
plt.ylabel('Sale price in U.S. dollars')
plt.show()

target = 'SalePrice'
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=123)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(
    n_estimators=1_000, criterion='squared_error', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.3f}')
print(f'MAE test: {mae_test:.3f}')
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.3f}')
print(f'R^2 test: {r2_test:.3f}')

plot_residual(X_train, y_train, X_test, y_test, forest)
