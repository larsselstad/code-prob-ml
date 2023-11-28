# source: https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_dtree.ipynb

# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# to make this notebook's output stable across runs
np.random.seed(42)


def plot_surface(clf, X, y, xnames, ynames):
    n_classes = 3
    plot_step = 0.02
    markers = ["o", "s", "^"]

    plt.figure(figsize=(10, 10))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.xlabel(xnames[0])
    plt.ylabel(xnames[1])

    # we pick a color map to match that used by decision tree graphviz
    cmap = ListedColormap(["orange", "green", "purple"])
    # cmap = ListedColormap(['blue', 'orange', 'green'])
    # cmap = ListedColormap(sns.color_palette())
    plot_colors = [cmap(i) for i in range(4)]

    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            label=ynames[i],
            edgecolor="black",
            color=color,
            s=50,
            cmap=cmap,
            marker=marker,
        )
    plt.legend()


iris = load_iris()
print(iris.target_names)
print(iris.feature_names)

# ndx = [0, 2] # sepal length, petal length
ndx = [2, 3]  # petal lenght and width
X = iris.data[:, ndx]
y = iris.target
xnames = [iris.feature_names[i] for i in ndx]
ynames = iris.target_names

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

plot_surface(tree_clf, X, y, xnames, ynames)
plt.savefig("figures/dtree_iris_depth2_surface.png")

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

plot_surface(tree_clf, X, y, xnames, ynames)
plt.savefig("figures/dtree_iris_depth3_surface.png")

tree_clf = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_clf.fit(X, y)

plot_surface(tree_clf, X, y, xnames, ynames)
plt.savefig("figures/dtree_iris_depth_unrestricted_surface.png")
