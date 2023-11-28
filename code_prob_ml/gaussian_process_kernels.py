# source: https://github.com/probml/pyprobml/blob/master/notebooks/book2/18/gpKernelPlot.ipynb

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import tinygp
import jaxopt
from tinygp import GaussianProcess, kernels
import numpy as np
import warnings
import os


class LinearKernel(tinygp.kernels.Kernel):
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, X1, X2):
        X1 = jnp.atleast_1d(X1)[..., None]
        X2 = jnp.atleast_1d(X2)[..., None]
        return self.scale * (jnp.sum(jnp.matmul((X1 + 1), (X2 + 1))))


class ConstantKernel(tinygp.kernels.Kernel):
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, X1, X2):
        X1 = jnp.atleast_1d(X1)[..., None]
        X2 = jnp.atleast_1d(X2)[..., None]
        return self.scale * jnp.sum(jnp.ones(jnp.shape(X1 * X2)))


class PolyLinearKernel(tinygp.kernels.Kernel):
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, X1, X2):
        X1 = jnp.atleast_1d(X1)[..., None]
        X2 = jnp.atleast_1d(X2)[..., None]
        return self.scale * (jnp.sum(1 + jnp.matmul(X1, X2)))


class WhiteNoiseKernel(tinygp.kernels.Kernel):
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, X1, X2):
        X1 = jnp.atleast_1d(X1)[..., None]
        X2 = jnp.atleast_1d(X2)[..., None]
        X3 = X1 == X2
        X4 = jnp.zeros_like(X1)
        X4 = jnp.where(X3, 1, X4)
        return self.scale * (jnp.sum(X4))


class QuadKernel(tinygp.kernels.Kernel):
    def __init__(self, scale):
        self.scale = scale

    def evaluate(self, X1, X2):
        X1 = jnp.atleast_1d(X1)[..., None]
        X2 = jnp.atleast_1d(X2)[..., None]
        return self.scale * (jnp.sum((1 + jnp.matmul(X1, X2))) ** 2)


def gp_kernel_plot(seed):
    n_xstar = 201
    num_examples = 3
    x_range = jnp.linspace(-10, 10, n_xstar).reshape(-1, 1)
    numerical_noise = 1e-5
    k = kernels.Matern52(scale=1.0)
    gp = GaussianProcess(k, x_range, diag=numerical_noise)
    no_grid_rows = 4
    no_grid_cols = 3

    label = np.array(
        [
            "(a)",
            "(b)",
            "(c)",
            "(d)",
            "(e)",
            "(f)",
            "(g)",
            "(h)",
            "(i)",
            "(j)",
            "(k)",
            "(l)",
        ]
    )
    se_kernel = kernels.ExpSquared(scale=1.0)
    lin_kernel = LinearKernel(scale=1.0)
    quad_kernel = QuadKernel(scale=1.0)
    matern_kernel_1 = kernels.Exp(scale=1.0)
    matern_kernel_3 = kernels.Matern32(scale=1.0)
    matern_kernel_5 = kernels.Matern52(scale=1.0)
    periodic_kernel = kernels.ExpSineSquared(scale=4.0, gamma=jnp.array(1.0))
    cosine_kernel = kernels.Cosine(scale=5.0)
    rational_quadratic_kernel = kernels.RationalQuadratic(scale=2, alpha=5)
    constant_kernel = ConstantKernel(scale=1)
    poly_linear_kernel = PolyLinearKernel(scale=0.5)
    white_noise_kernel = WhiteNoiseKernel(scale=1)

    kernel_names = {
        "Matern12": (matern_kernel_1, 0.0),
        "Matern32": (matern_kernel_3, 0.0),
        "Matern52": (matern_kernel_5, 0.0),
        "Periodic": (periodic_kernel, 0.0),
        "Cosine": (cosine_kernel, 0.0),
        "RBF": (se_kernel, 0.0),
        "Rational quadratic": (rational_quadratic_kernel, 0.0),
        "Constant": (constant_kernel, 0.0),
        "Linear": (lin_kernel, 1.0),
        "Quadratic": (quad_kernel, 1.0),
        "Polynomial": (poly_linear_kernel, 1.0),
        "White noise": (white_noise_kernel, 0.0),
    }

    fig, axs = plt.subplots(
        nrows=no_grid_rows, ncols=no_grid_cols, constrained_layout=True
    )
    for fig_no, (kernel_name, ax) in enumerate(zip(kernel_names, axs.ravel())):
        kernel = kernel_names[kernel_name][0]

        gp = GaussianProcess(kernel, x_range, diag=numerical_noise)
        samples = jnp.atleast_2d(
            gp.sample(jax.random.PRNGKey(seed), shape=(num_examples,)).T
        )
        ax.plot(x_range, samples, alpha=0.8)
        ax.set_xlabel(label[fig_no] + " " + kernel_name)
    sns.despine()
    # plt.show()
    plt.savefig("figures/gaussian_process_kernel/gpKernelSamples_latexified.png")

    fig, axs = plt.subplots(
        nrows=no_grid_rows, ncols=no_grid_cols, constrained_layout=True
    )

    for fig_no, (kernel_name, ax) in enumerate(zip(kernel_names, axs.ravel())):
        kernel = kernel_names[kernel_name][0]
        X1 = (
            jnp.array([[0.0]])
            if kernel_names[kernel_name][1] == 0.0
            else jnp.array([[1.0]])
        )
        ax.plot(x_range, kernel(x_range, X1))
        ax.set_xlabel(
            label[fig_no] + " " + f"{kernel_name} k(x,{kernel_names[kernel_name][1]})"
        )
    sns.despine()
    # plt.show()
    plt.savefig("figures/gaussian_process_kernel/gpKernels_latexified.png")


gp_kernel_plot(15)
