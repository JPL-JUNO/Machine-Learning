"""
@Description: Grouping objects by similarity using k-means
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-26 09:55:02
"""

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, n_features=2,
                  centers=3, cluster_std=.5, shuffle=True, random_state=0)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.grid()
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random', n_init=10, max_iter=300, tol=1e-4,
            random_state=0)
y_km = km.fit_predict(X)
print(f'Distortion: {km.inertia_:.2f}')

distortions = []
num = 11
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++', n_init=10, max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, num), distortions, marker='o')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Distortion')
plt.tight_layout()
plt.show()


km = KMeans(n_clusters=3,
            init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)
y_km = km.fit_transform(X)

import numpy as np
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(
    X, y_km, metric='euclidean'
)
y_ax_lower, y_ax_upper = 0, 0
for i, c in enumerate(cluster_labels):
    pass
