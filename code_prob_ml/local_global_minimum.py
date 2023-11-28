# source: https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/08/extrema_fig_1d.ipynb
# Plot local minimum and maximum in 1d
# https://nbviewer.jupyter.org/github/entiretydotai/Meetup-Content/blob/master/Neural_Network/7_Optimizers.ipynb


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def f(x):
    return x * np.sin(-np.pi * x)


a = np.arange(-1, 3, 0.01)
plt.annotate(
    "local minimum",
    xy=(0.7, -0.55),
    xytext=(0.1, -2.0),
    arrowprops=dict(facecolor="black"),
)

plt.annotate(
    "Global minimum",
    xy=(2.5, -2.5),
    xytext=(0.1, -2.5),
    arrowprops=dict(facecolor="black"),
)

plt.plot(a, f(a))
plt.savefig("figures/local_global_minimum/extrema_fig_1d.png")
