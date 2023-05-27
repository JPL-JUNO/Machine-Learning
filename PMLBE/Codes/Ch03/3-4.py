"""
@Description: Classifying face images with SVM
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-26 22:32:29
"""


from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# 每个人最少要有80张图片
face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target
assert X.shape == (1140, 2914)
assert Y.shape == (1140,)
print('Label names: {}'.format(face_data.target_names))

for i in range(5):
    print('Class {0} has {1} samples'.format(i, (Y==i).sum()))
    
fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.ravel()):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=face_data.target_names[face_data.target[i]])
plt.tight_layout()
plt.show()

from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, random_state=42)
clf = SVC(class_weight='balanced', random_state=42)
parameters ={'C':[.1,1,10],
             'gamma': [1e-07, 1e-08, 1e-06],
             'kernel': ['rbf', 'linear']}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)
print('The best model: \n', grid_search.best_params_) 
print('The best averaged performance:', grid_search.best_score_)

clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)
from sklearn.metrics import classification_report
print('The accuracy is: {0:.1f}%'.format(clf_best.score(X_test, Y_test)*100))
print(classification_report(Y_test, pred, target_names=face_data.target_names))

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)
model = Pipeline([('pca', pca),
                  ('svc', svc)])

parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [.001, .005]}
grid_search = GridSearchCV(model, parameters_pipeline)
grid_search.fit(X_train, Y_train)
print('The best model:\n', grid_search.best_params_)
print('The best averaged performance:\n', grid_search.best_score_)
model_best = grid_search.best_estimator_
print('The accuracy is: {0:.2f}%'.format(model_best.score(X_test, Y_test)*100))
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))