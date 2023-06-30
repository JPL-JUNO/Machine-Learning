"""
@Description: Classifying handwritten digits
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-30 15:41:56
"""

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, parser='auto',
                    return_X_y=True)
X = X.values
y = y.astype(int).values
assert X.shape == (70_000, 784)
assert y.shape == (70_000,)
# gradient-based optimization is much more stable under these conditions
X = 2 * (X / 255 - .5)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    img = X[y == i][0].reshape(28, 28)
    ax.imshow(img, cmap='Greys')
axes[0][0].set_xticks([])
axes[1][0].set_yticks([])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    img = X[y == 7][i].reshape(28, 28)
    ax.imshow(img, cmap='Greys')
axes[0][0].set_xticks([])
axes[1][0].set_yticks([])
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10_000,
                                                  random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5_000, random_state=123, stratify=y)
