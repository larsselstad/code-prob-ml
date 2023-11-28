# source: https://github.com/probml/pyprobml/blob/master/notebooks/book2/18/gpr_demo_change_hparams.ipynb

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import tinygp

kernels = tinygp.kernels
from tinygp import GaussianProcess

# latexify(width_scale_factor=2, fig_height=1.75)

marksize = 30

# train data
n = 20
key = jax.random.PRNGKey(0)
key_split = jax.random.split(key, 2)

x = 15 * (jax.random.uniform(key, shape=(n,), minval=0, maxval=1) - 0.5).reshape((-1,))

sigma_y = 0.1
sigma_f = 1.0
length_scale = 1.0
kernel = (sigma_f**2) * kernels.ExpSquared(scale=length_scale)
gp = GaussianProcess(kernel, x, diag=sigma_y**2)
y = gp.sample(key_split[0], shape=(1,)).T
y = y.reshape(
    -1,
)

# test data
xtest = jnp.linspace(-10, 10, 201).reshape(-1)


def generate_plots(sigma_f, length_scale, sigma_y):
    kernel = (sigma_f**2) * kernels.ExpSquared(scale=length_scale)
    gp = GaussianProcess(kernel, x, diag=sigma_y**2)
    cond_gp = gp.condition(y, xtest).gp
    mu, var = cond_gp.loc, cond_gp.variance

    plt.plot(xtest, mu, "-")
    plt.scatter(x, y, c="k", s=marksize, marker="x")
    plt.fill_between(
        xtest.flatten(),
        mu.flatten() - 1.96 * jnp.sqrt(var),
        mu.flatten() + 1.96 * jnp.sqrt(var),
        alpha=0.3,
    )
    sns.despine()
    plt.legend(
        labels=["Mean", "Data", "Confidence"], loc=2, prop={"size": 4.5}, frameon=False
    )
    plt.title(
        "$(l, \sigma_f, \sigma_y)=${}, {}, {}".format(length_scale, sigma_f, sigma_y)
    )
    plt.xlabel("$x$"), plt.ylabel("$f$")
    plt.yticks(jnp.linspace(-2, 4, 4))
    plt.savefig(f"figures/rbf_kernels/gprDemoChangeHparams{i}.png")


lengthscale_array = jnp.array([1.0, 0.3, 3.0])
sigmaf_array = jnp.array([1, 1.08, 1.16])
sigmay_array = jnp.array([0.1, 0.00005, 0.89])
for i in range(len(lengthscale_array)):
    plt.figure(i)
    generate_plots(sigmaf_array[i], lengthscale_array[i], sigmay_array[i])
