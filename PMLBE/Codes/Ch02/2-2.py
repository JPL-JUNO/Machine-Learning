"""
@Description: Building a movie recommender with Naïve Bayes
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-25 09:36:05
"""

import numpy as np
from collections import defaultdict
from typing import Tuple
from numpy import ndarray

data_path = 'ml-1m/ratings.dat'
n_users = 6040
n_movies = 3706


def load_rating_data(data_path: str, n_users: int, n_movies: int) -> Tuple[ndarray, dict, dict]:
    """加载文件数据

    Args:
        data_path (str): 文件路径
        n_users (int): 用户数量
        n_movies (int): 电影数量

    Returns:
        Tuple[ndarray, dict, dict]: 由数据、{电影ID:评分熟练}、{电影ID:}
    """
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split('::')
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


data, movie_n_rating, movie_id_mapping = load_rating_data(
    data_path, n_users, n_movies)


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print('Number of rating {0}: {1}'.format(int(value), count))


movie_id_most, n_rating_most = sorted(movie_n_rating.items(),
                                      key=lambda d: d[1], reverse=True)[0]
print('Movie ID {} has {} ratings.'.format(movie_id_most, n_rating_most))

X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

assert X.shape == (3428, 3705)
assert Y.shape == (3428,)


display_distribution(Y)
recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()

print('{0} positive samples and {1} negative samples'.format(n_pos, n_neg))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=42)
assert len(Y_train) == 2742
assert len(Y_test) == 686

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

prediction_prob = clf.predict_proba(X_test)
prediction = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)
print('The accuracy is {0:.1f} %'.format(accuracy * 100))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0, 1]))

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)

f1_score(Y_test, prediction, pos_label=0)

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)


pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, .05)
true_pos, false_pos = [0] * len(thresholds), [0] * len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:  # 会被预测为1
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:  # 被预测为0的不考虑，直接换下一个样本
            break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()

true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, pos_prob)


from sklearn.model_selection import StratifiedKFold
k = 5
k_fold = StratifiedKFold(n_splits=k)
smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train, Y_train)
            prediction_prob = clf.predict_proba(X_test)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test, pos_prob)
            auc_record[alpha][fit_prior] = auc + \
                auc_record[alpha].get(fit_prior, 0.0)

for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print('{0} {1} {2:.5f}'.format(smoothing, fit_prior, auc / k))

clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('Auc with the best model: {}'.format(roc_auc_score(Y_test, pos_prob)))
