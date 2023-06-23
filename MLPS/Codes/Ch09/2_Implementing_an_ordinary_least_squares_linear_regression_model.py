"""
@Description: Implementing an ordinary least squares linear regression model
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-23 20:51:19
"""

import sys
sys.path.append('./')
sys.path.append('../')

from modelsNN.LinearRegressionGD import LinearRegressionGD
from utilsML.visualize import lin_reg_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('AmesHousing.txt', sep='\t', usecols=columns)
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
# Most data preprocessing classes in scikit-learn expect data to be stored in two-dimensional arrays
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD(eta=.1)
lr.fit(X_std, y_std)

fig, ax = plt.subplots()
ax.plot(range(lr.n_iter), lr.losses_)
ax.set_ylabel('MSE')
ax.set_xlabel('Epoch')
plt.show()


fig, ax = plt.subplots()
lin_reg_plot(X_std, y_std, lr, ax)
ax.set_xlabel('Living area above ground (standardized)')
ax.set_ylabel('Sale price (standardized)')
plt.show()


feature_std = sc_x.transform(np.array([[2_500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f'Sales price: ${target_reverted.flatten()[0]:.2f}')


print(f'Slope: {lr.w_[0]:.3f}')
print(f'Intercept: {lr.b_[0]:.3f}')


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print(f'Slope: {slr.coef_[0]:.3f}')
print(f'Intercept: {slr.intercept_:.3f}')

fig, ax = plt.subplots()
lin_reg_plot(X, y, slr, ax)
ax.set_xlabel('Living area above ground in square feet')
ax.set_ylabel('Sale price in U.S. dollars')
plt.tight_layout()
plt.show()


Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print(f'Slope: {w[1]:.3f}')
print(f'Intercept: {w[0]:.3f}')
