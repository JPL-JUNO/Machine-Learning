"""
@Description: Handling multiclass classification
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-30 16:30:34
"""

from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
digits = load_digits()
n_samples = len(digits.images)

X = digits.images.reshape((n_samples, -1))
Y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=42)
from sklearn.model_selection import GridSearchCV
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10]}
sgd_lr = SGDClassifier(loss='log_loss', learning_rate='constant',
                       eta0=.01, fit_intercept=True, max_iter=30)
grid_search = GridSearchCV(sgd_lr, parameters, n_jobs=-1, cv=3)
grid_search.fit(X_train, Y_train)
sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(X_test, Y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy * 100))
