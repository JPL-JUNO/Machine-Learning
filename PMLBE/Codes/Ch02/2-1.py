"""
@Description: Implementing Naïve Bayes
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-24 21:00:51
"""

import numpy as np
from typing import Dict
from numpy import ndarray
X_train = np.array([[0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0],
                    [1, 1, 0]])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[1, 1, 0]])


def get_label_indices(labels: list) -> Dict:
    """Group samples based on their labels and return indices

    Args:
        labels (list): list of labels

    Returns:
        Dict[list]: {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


def get_prior(label_indices: dict) -> dict:
    """Compute prior based on training samples

    Args:
        label_indices (dict): grouped sample indices by class

    Returns:
        dict: with class label as key, corresponding prior as the value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


def get_likelihood(features, label_indices, smoothing: int = 0):
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / \
            (total_count + len(label_indices) * smoothing)
    return likelihood


def get_posterior(X: ndarray, prior: dict, likelihood: dict) -> list:
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (
                    1 - likelihood_label[index])
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


label_indices = get_label_indices(Y_train)
prior = get_prior(label_indices)
likelihood = get_likelihood(X_train, label_indices, 1)
posterior = get_posterior(X_test, prior, likelihood)

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities: \n', pred_prob)

pred = clf.predict(X_test)
print('[scikit-learn] Prediction: ', pred)
