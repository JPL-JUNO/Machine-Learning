"""
@Description: Bagging – building an ensemble of classifiers from bootstrap samples
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-15 09:52:04
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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier(
    criterion='entropy', random_state=1, max_depth=None)
bag = BaggingClassifier(estimator=tree,
                        n_estimators=500, max_samples=1.0, max_features=1.0,
                        bootstrap_features=False, n_jobs=-1, random_state=1)
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'DecisionTree train/test accuracies {tree_train:.3f}/{tree_test:.3f}')

bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print(f'Bagging train/test accuracies {bag_train:.3f}/{bag_test:.3f}')
from utilsML.visualize import plot_decision_regions_subplots
plot_decision_regions_subplots(X_train, y_train, classifier=[
                               tree, bag], n_rows=1, n_cols=2, title=['Decision tree', 'Bagging'],
                               xylabel=['OD280/OD315 of diluted wines', 'Alcohol'])
plt.show()
