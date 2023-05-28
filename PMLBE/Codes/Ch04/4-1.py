"""
@Description: The metrics for measuring a split
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-28 15:36:19
"""

# In binary cases, Gini Impurity, under different values of the positive class fraction,
# can be visualized by the following code blocks:
import matplotlib.pyplot as plt
import numpy as np

pos_fraction = np.linspace(0, 1, 1000)
gini = 1 - pos_fraction**2 - (1 - pos_fraction)**2
plt.plot(pos_fraction, gini, label='Gini impurity')
plt.ylim(0, 1.05)
plt.xlabel('Positive fraction')


def gini_impurity(labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


pos_fraction = np.linspace(0.00001, .99999, 1000)
ent = -pos_fraction * np.log2(pos_fraction) - \
    (1 - pos_fraction) * np.log2(1 - pos_fraction)

plt.plot(pos_fraction, ent, label='Entropy')
plt.ylabel('Values')
plt.legend()


def entropy(labels):
    if not labels:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / len(labels)
    return -np.sum(fractions * np.log2(fractions))


criterion_function = {'gini': gini_impurity,
                      'entropy': entropy}


def weighted_impurity(groups, criterion: str = 'gini'):
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * \
            criterion_function[criterion](group)
    return weighted_sum
