"""
@Description: Estimating with support vector regression
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-01 10:18:23
"""

from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
diabetes = load_diabetes()
num_test = 10
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:,]
y_test = diabetes.target[-num_test:]

regressor = SVR(C=.1, epsilon=.02, kernel='linear')
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
