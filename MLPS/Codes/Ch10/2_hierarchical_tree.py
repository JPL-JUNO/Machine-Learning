"""
@Description: Organizing clusters as a hierarchical tree
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-28 20:34:38
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
sys.path.append('../')
from utilsML.visualize import plot_cluster
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.standard_normal(size=(5, 3)) * 10
df = pd.DataFrame(data=X, columns=variables, index=labels)

from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(
    pdist(df, metric='euclidean')), columns=labels, index=labels)

from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
row_clusters = linkage(df.values, method='complete', metric='euclidean')
# pd.DataFrame(row_clusters,
#              columns=['row label 1',
#                       'row label 2',
#                       'distance',
#                       'no. of items in clust.'],
#              index=[f'cluster {(i + 1)}' for i in
#                     range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.show()

# Attaching dendrograms to a heat map
fig = plt.figure(figsize=(7, 7), facecolor='white')
axd = fig.add_axes([.09, .1, .2, .6])
row_dendr = dendrogram(row_clusters, orientation='left')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([.23, .1, .6, .6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest', cmap='hot_r')
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(
    n_clusters=3, metric='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')


ac = AgglomerativeClustering(
    n_clusters=2, metric='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,
            random_state=0)
y_km = km.fit_predict(X)
plot_cluster(X, km, 2, plot_centers=False)
ac = AgglomerativeClustering(
    n_clusters=2, metric='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)
plot_cluster(X, ac, 2, plot_centers=False)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
plot_cluster(X, db, 2, plot_centers=False)
