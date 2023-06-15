"""
@Description: Applying AdaBoost using scikit-learn
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-15 19:59:20
"""
import pandas as pd
import sys
sys.path.append('./')
sys.path.append('../')
import matplotlib.pyplot as plt
from data.get_data import DataLoader
dl = DataLoader()
df_wine = dl.get_wine()

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash',
                   'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1, stratify=y)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1, max_depth=1)
ada = AdaBoostClassifier(estimator=tree, n_estimators=500,
                         learning_rate=.1, random_state=1)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}')

ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f'AdaBoost train/test accuracies {ada_train:.3f}/{ada_test:.3f}')

from utilsML.visualize import plot_decision_regions_subplots
plot_decision_regions_subplots(X_train, y_train, classifier=[tree, ada],
                               n_cols=2, n_rows=1, title=['tree', 'AdaBoost'],
                               xylabel=[
                                   'OD280/OD315 of diluted wines', 'Alcohol'],
                               tight_layout=True)
