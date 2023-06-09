"""
@Description: Scenario 4 – dealing with more than two classes
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-26 15:53:22
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
wine_data = load_wine()
X = wine_data.data
Y = wine_data.target

assert X.shape == (178, 13)
assert Y.shape == (178,)

print('Label name: ', wine_data.target_names)
n_class_0 = (Y == 0).sum()
n_class_1 = (Y == 1).sum()
n_class_2 = (Y == 2).sum()

assert n_class_0 == 59
assert n_class_1 == 71
assert n_class_2 == 48

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print('The accuracy is: {0:.1f}%'.format(accuracy * 100))
pred = clf.predict(X_test)
print(classification_report(Y_test, pred))
