"""
@Description: Implementing SVM
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-26 09:02:47
"""


from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target
assert X.shape == (569, 30)
assert Y.shape == (569,)

assert all(cancer_data.target_names == ['malignant', 'benign'])

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()

assert n_pos == 357
assert n_neg == 212

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print('The accuracy is {0:.1f}%'.format(accuracy * 100))
