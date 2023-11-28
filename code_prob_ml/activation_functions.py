# source: https://github.com/probml/pyprobml/blob/master/notebooks/book1/13/activation_fun_deriv.ipynb

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

x = jnp.arange(-4.0, 4.0, 0.1)
partial_leaky_relu = lambda x: jax.nn.leaky_relu(x, negative_slope=0.1)

fns = [
    jax.nn.sigmoid,
    # jax.nn.relu,
    partial_leaky_relu,
    jax.nn.elu,
    jax.nn.swish,
    jax.nn.gelu,
]

names = [
    "sigmoid",
    #'relu',
    "leaky-relu",
    "elu",
    "swish",
    "gelu",
]

# evaluate functions and their gradients on a grid of points
fdict, gdict = {}, {}

for fn, name in zip(fns, names):
    y = fn(x)
    grad_fn = jax.grad(lambda x: fn(x))
    grads = jax.vmap(grad_fn)(x)

    fdict[name] = y  # vector of fun
    gdict[name] = grads  # gradient wrt x(i)

# Plot the funcitons
styles = ["r-", "g--", "b-.", "m:", "k-"]
ax = plt.subplot()
for i, name in enumerate(names):
    lab = f"{name}"
    ax.plot(x, fdict[name], styles[i], linewidth=0.7, label=lab)
ax.set_ylim(-0.5, 2)
ax.legend(frameon=False, loc="upper left", borderaxespad=0.1, labelspacing=0.2)
sns.despine()
plt.title("Activation function")
# not saving grafh as image because I don't need it
# plt.savefig("activation-funs")
plt.show()

ax = plt.subplot()
for i, name in enumerate(names):
    lab = f"{name}"
    ax.plot(x, gdict[name], styles[i], linewidth=0.7, label=lab)
ax.set_ylim(-0.5, 2)
ax.legend(frameon=False, loc="upper left", borderaxespad=0.1, labelspacing=0.2)
sns.despine()
plt.title("Gradient of activation function")
# not saving grafh as image because I don't need it
# plt.savefig("activation-funs-grad")
plt.show()
