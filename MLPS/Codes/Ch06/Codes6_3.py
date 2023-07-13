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
ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度移到下面
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
param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
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

from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', C=100)
)
X_train_2 = X_train[:, [4, 14]]
from sklearn.model_selection import StratifiedKFold
cv = list(StratifiedKFold(n_splits=3).split(X_train_2, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv, 1):
    probas = pipe_lr.fit(
        X_train_2[train], y_train[train]).predict_log_proba(X_train_2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1], pos_label=1)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC fold {i}(area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color=(.6, .6, .6),
         label='Random guessing (area=.5)')
mean_tpr /= len(cv)
mean_tpr[-1] = 1
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (area = {mean_auc:.2f})')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':',
         color='black', label='Perfect performance (area = 1.0)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1, greater_is_better=True, average='micro')
# Dealing with class imbalance
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100


from sklearn.utils import resample
print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])
X_upsmapled, y_upsampled = resample(
    X_imb[y_imb == 1],
    y_imb[y_imb == 1], replace=True, n_samples=X_imb[y_imb == 0].shape[0], random_state=123
)
print('Number of class 1 examples after:', X_upsmapled.shape[0])

X_bal = np.vstack((X[y == 0], X_upsmapled))
y_bal = np.hstack((y[y == 0], y_upsampled))
y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100
