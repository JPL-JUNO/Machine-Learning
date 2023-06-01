"""
@Description: Evaluating regression performance
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-01 10:41:27
"""
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
diabetes = load_diabetes()
num_test = 30
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:,]
y_test = diabetes.target[-num_test:]

param_grid = {
    'alpha': [1e-07, 1e-06, 1e-05],
    'penalty': [None, 'l2'],
    'eta0': [.03, .05, .1],
    'max_iter': [500, 1_000]
}
from sklearn.model_selection import GridSearchCV
regressor = SGDRegressor(loss='squared_error',
                         learning_rate='constant', random_state=42)
grid_search = GridSearchCV(regressor, param_grid)
grid_search.fit(X_train, y_train)
regressor_best = grid_search.best_estimator_
predictions = regressor_best.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
