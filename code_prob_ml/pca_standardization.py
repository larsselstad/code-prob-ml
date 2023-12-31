# source: https://github.com/probml/pyprobml/blob/master/notebooks/book1/20/pcaStandardization.ipynb


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

import os
import scipy.io

# Get Data
data = scipy.io.loadmat("code_prob_ml/heightWeight.mat")["heightWeightData"]
data = pd.DataFrame(data).rename(columns={0: "gender", 1: "height", 2: "weight"})


# Function for plotting categorical scatter plot with 1D PCA line
def make_pca_plot(data):
    pca = PCA(1)
    X_reconstr = pca.inverse_transform(
        pca.fit_transform(data[["height", "weight"]].values)
    )
    X_reconstr = np.sort(X_reconstr, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (name, group) in enumerate(data.groupby("gender")):
        color = "red" if i == 1 else "blue"
        marker = "o" if i == 1 else "x"
        fc = "none" if i == 1 else "blue"
        ax.scatter(
            x=group["height"],
            y=group["weight"],
            color=color,
            marker=marker,
            facecolor=fc,
            s=100,
        )
        ax.set_ylabel("weight")
        ax.set_xlabel("height")
        ax.plot(X_reconstr[:, 0], X_reconstr[:, 1], color="black", linewidth=2)

    return fig, ax


# Create and save figures
fig, ax = make_pca_plot(data)
ax.set_title("heightWeightPCA")
plt.savefig("figures/pca_standardization/heightWeightPCA.png")

data_std = (data - data.mean()) / data.std()
fig, ax = make_pca_plot(data_std)
ax.set_title("heightWeightPCAstnd")
plt.savefig("figures/pca_standardization/heightWeightPCAstnd.png")
