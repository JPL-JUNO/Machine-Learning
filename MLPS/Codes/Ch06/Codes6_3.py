"""
@Description: Looking at different performance evaluation metrics
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-13 20:44:39
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

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(x=col, y=row, s=cm[row, col], va='center', ha='center')
ax.xaxis.set_ticks_position('bottom') # 将x轴刻度移到下面
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

pre_val = precision_score(y_true=y_test, y_pred=y_pred)
rec_val = recall_score(y_true=y_test, y_pred=y_pred)
f1_val = f1_score(y_true=y_test, y_pred=y_pred)
mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')
print(f'Recall: {rec_val:.3f}')
print(f'F1: {f1_val:.3f}')
print(f'MCC: {mcc_val:.3f}')

from sklearn.metrics import make_scorer
c_gamma_range = [.01, .1, 1, 10.0]
param_grid = [{'svc__C':c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C':c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}
              ]
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)