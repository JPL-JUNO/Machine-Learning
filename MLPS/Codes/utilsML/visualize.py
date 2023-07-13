"""
@Description: plot_decision_regions
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-03 14:09:38
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as Patheffects
from array import array
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import Axes
from numpy import ndarray
from itertools import product
from matplotlib import cm


def plot_decision_regions(X, y, classifier, resolution: float = .02, test_idx=None):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    _ = plt.figure()
    plt.contourf(xx1, xx2, lab, alpha=.3, cmap=cmap)
    plt.axis([x1_min, x1_max, x2_min, x2_max])

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1], alpha=.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}', edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=1, c='none', edgecolors='black',
                    linewidth=1, marker='o', s=100, label='Test samples')


def plot_project(x: ndarray, colors: array):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])
    for i in range(10):
        x_text, y_text = np.median(x[colors == i, :], axis=0)
        txt = ax.text(x_text, y_text, str(i), fontsize=24)
        txt.set_path_effects([Patheffects.Stroke(linewidth=5, foreground='w'),
                              Patheffects.Normal()])


def plot_decision_regions_subplots(X, y, classifier,
                                   n_cols: int, n_rows: int, title: list,
                                   xylabel: list,
                                   tight_layout: bool = True,
                                   resolution: float = .02):
    if X.shape[1] != 2:
        raise ValueError(f"X must be has 2 columns, got {X.shape[1]}")
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    fig, axarr = plt.subplots(ncols=n_cols, nrows=n_rows,
                              sharex='col', sharey='row', figsize=(n_cols * 3, n_rows * 3))
    for ax, clf, tt in zip(axarr.ravel(),
                           classifier, title):
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='^')
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='green', marker='o')
        ax.set_title(tt)
    fig.supxlabel(xylabel[0], fontsize=12)
    fig.supylabel(xylabel[1], fontsize=12)
    if tight_layout:
        plt.tight_layout()


def lin_reg_plot(X, y, model, ax: Axes):
    ax.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    ax.plot(X, model.predict(X), color='black', lw=2)


def plot_residual(X_train, y_train, X_test, y_test, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    ax1.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    ax1.set_ylabel('Residuals')
    for ax in (ax1, ax2):
        ax.set_xlabel('Predicted values')
        ax.legend()
        ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100,
                  color='black', lw=2)
    plt.tight_layout()
    plt.show()


def plot_k_means(X, model, centers: int = 3):
    y_km = model.fit_predict(X)
    shapes = ['s', 'v', '^', 'o']
    colors = ['lightgreen', 'orange', 'lightblue', 'lightyellow']
    fig, ax = plt.subplots()
    for i, shape, color in zip(range(centers), shapes, colors):
        ax.scatter(X[y_km == i, 0],
                   X[y_km == i, 1], s=50, c=color, marker=shape,
                   label=f'Cluster {i}')

    ax.scatter(model.cluster_centers_[:, 0],
               model.cluster_centers_[:, 1],
               s=250, marker='*', c='red', edgecolor='black', label='Centroids')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_cluster(X, model, centers: int = 3, plot_centers: bool = True):
    y_km = model.fit_predict(X)
    shapes = ['s', 'v', '^', 'o']
    colors = ['lightgreen', 'orange', 'lightblue', 'lightyellow']
    fig, ax = plt.subplots()
    for i, shape, color in zip(range(centers), shapes, colors):
        ax.scatter(X[y_km == i, 0],
                   X[y_km == i, 1], s=50, c=color, marker=shape,
                   label=f'Cluster {i}')
    if plot_centers:
        ax.scatter(model.cluster_centers_[:, 0],
                   model.cluster_centers_[:, 1],
                   s=250, marker='*', c='red', edgecolor='black', label='Centroids')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_silhouette_coefficients(clusters, silhouette_vals, y_km):
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    n_clusters = clusters.shape[0]
    fig, ax = plt.subplots(1)
    for i, c in enumerate(clusters):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(float(i) / n_clusters)
        ax.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals,
                height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    ax.axvline(silhouette_avg, color='red', linestyle='--')
    ax.set_yticks(yticks, clusters + 1)
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()


def plot_digit_number(nrows: int = 2, ncols: int = 5, sharex: bool = True, sharey: bool = True):
    pass
    fig, axes = plt.subplots(nrows, ncols, sharex, sharey)
    for i, ax in enumerate(axes.ravel()):
        img = X[y == i][0].reshape(28, 28)
    pass


if __name__ == '__main__':
    pass
