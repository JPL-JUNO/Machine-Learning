"""
@Description: Fine-tuning machine learning models via grid search
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 09:31:47
"""

import sys

import numpy as np
sys.path.append('./')
sys.path.append('../')
from data.get_data import DataLoader
dl = DataLoader()
X, y = dl.get_wdbc()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    stratify=y, random_state=1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
param_range = [.0001, .001, .01, .1, 1, 10, 100, 1000]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')
print(f'Test accuracy: {gs.score(X_test, y_test):.3f}')


import scipy
param_range = scipy.stats.loguniform(.0001, 1000)
np.random.seed(1)
param_range.rvs(10)

from sklearn.model_selection import RandomizedSearchCV
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))
rs = RandomizedSearchCV(estimator=pipe_svc, param_distributions=param_grid,
                        scoring='accuracy',refit=True, n_iter=20, cv=10, 
                        random_state=1, n_jobs=-1)
rs.fit(X_train, y_train)
print(rs.best_score_)
print(rs.best_params_)

# More resource-efficient hyperparameter search with successive halving
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
hs = HalvingRandomSearchCV(pipe_svc,
                           param_distributions=param_grid,
                           n_candidates='exhaust',
                           resource='n_samples',
                           factor=1.5, random_state=1, n_jobs=-1)
hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)
clf = hs.best_estimator_
print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')

# Algorithm selection with nested cross-validation
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,scoring='accuracy',cv=2)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gs, X_train, y_train, 
                         scoring='accuracy',cv=5)
print(f'CV Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# compare an svm model to a simple decision tree
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print(f'CV Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')