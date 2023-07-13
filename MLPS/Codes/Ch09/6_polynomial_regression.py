"""
@Description: Turning a linear regression model into a curve – polynomial regression
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-24 21:42:45
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

X = np.array([ 258.0, 270.0, 294.0, 320.0, 342.0,
              368.0, 396.0, 446.0, 480.0, 586.0,])[:, np.newaxis]

y = np.array([ 236.4, 234.4, 252.8, 298.6, 314.2, 
              342.2, 360.8, 368.0, 391.2, 390.8,])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
# y_quad_fit = pr.predict(quadratic.transform(X_fit))

fig, ax = plt.subplots()
ax.scatter(X, y, label='Training points')
ax.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
ax.plot(X_fit, y_quad_fit, label='Quadratic fit')
ax.set_xlabel('Explanatory variable')
ax.set_ylabel('Predicted or known target values')
plt.legend()
plt.tight_layout()
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
mse_lin = mean_squared_error(y, y_lin_pred)
mse_quad = mean_squared_error(y, y_quad_pred)
r2_lin = r2_score(y, y_lin_pred)
r2_quad  = r2_score(y, y_quad_pred)
print(f'Training MSE Linear: {mse_lin:.3f}, quadratic: {mse_quad:.3f}')
print(f'Training R2 Linear: {r2_lin:.3f}, quadratic: {r2_quad:.3f}')

# Modeling nonlinear relationships in the Ames Housing dataset
import pandas as pd
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('AmesHousing.txt', sep='\t', usecols=columns)
df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})
df = df.dropna(axis=0)

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values
mask = df['Gr Liv Area'] < 4_000
X = X[mask]
y = y[mask]

regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)


X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]
regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

fig, ax = plt.subplots()
ax.scatter(X, y, label='Training points', color='lightgray')
ax.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
        color='blue', lw=2, linestyle=':')
ax.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
        color='red', lw=2, linestyle='-')
ax.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
        color='green', lw=2, linestyle='--')
ax.set_xlabel('Living area above ground in square feet')
ax.set_ylabel('Sale price in U.S. dollars')
plt.legend()
plt.show()


X = df[['Overall Qual']].values
y = df['SalePrice'].values
# mask = df['Gr Liv Area'] < 4_000
# X = X[mask]
# y = y[mask]

regr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]
regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

fig, ax = plt.subplots()
ax.scatter(X, y, label='Training points', color='lightgray')
ax.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
        color='blue', lw=2, linestyle=':')
ax.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
        color='red', lw=2, linestyle='-')
ax.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
        color='green', lw=2, linestyle='--')
ax.set_xlabel('Living area above ground in square feet')
ax.set_ylabel('Sale price in U.S. dollars')
plt.legend()
plt.show()