"""
@Description: Decision tree Learning
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-06 10:23:48
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
sys.path.append('../')
from utilsML.visualize import plot_decision_regions


def entropy(p: float) -> float:
    """给定概率计算熵值（二分类）

    Args:
        p (float): 指定的概率

    Returns:
        float: 对应的熵值
    """
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


x = np.arange(0, 1, .01)
ent = [entropy(p) if p != 0 and p != 1 else None for p in x]
plt.ylabel('Entropy(binary class)')
plt.xlabel('Class-membership probability p(i=1)')
plt.plot(x, ent)
plt.show()


def gini(p: float):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def error(p: float):
    return 1 - np.max([p, 1 - p])


sc_ent = [e * .5 if e else None for e in ent]
err = [error(p) for p in x]
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gini(x), err],
                         ['Entropy', 'Entropy (scaled)',
                          'Gini impurity',
                          'Misclassification error'],
                         ['-', '-', '--', '-.'],
                         ['black', 'lightgray',
                          'red', 'green']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=5)
plt.ylim([0, 1.1])
ax.axhline(y=1, linewidth=1, color='k', linestyle='--')
ax.axhline(y=.5, linewidth=1, color='k', linestyle='--')
plt.xlabel('p(class=1)')
plt.ylabel('impurity index')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from codes1 import X_train, y_train, X_test, y_test
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

tree_model = DecisionTreeClassifier(
    criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=tree_model, test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from sklearn import tree
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
# Setting filled=True in the plot_tree function
# we called colors the nodes by the majority class label
# at that node.
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=25,
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


from codes1 import X_train_std, X_combined_std
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
