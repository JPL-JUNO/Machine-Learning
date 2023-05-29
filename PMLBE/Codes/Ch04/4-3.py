"""
@Description: Implementing a decision tree with scikit-learn
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 10:35:37
"""

from sklearn.tree import DecisionTreeClassifier, export_graphviz
X_train_n = [[6, 7],
             [2, 4],
             [7, 2],
             [3, 6],
             [4, 7],
             [5, 2],
             [1, 6],
             [2, 0],
             [6, 3],
             [4, 1]]
y_train_n = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
tree_sk = DecisionTreeClassifier(
    criterion='gini', max_depth=2, min_samples_split=2)
tree_sk.fit(X_train_n, y_train_n)
export_graphviz(tree_sk, out_file='tree.dot', feature_names=[
                'X1', 'X2'], impurity=False, filled=True, class_names=['0', '1'])
# $ dot -Tpng tree.dot -o tree.png
