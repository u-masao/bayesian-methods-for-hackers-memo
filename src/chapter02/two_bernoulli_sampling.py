import logging

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

from src.utils import save_trace_and_model


def sampling(occurences_a, occurences_b):
    model = pm.Model()
    with model:
        p_a = pm.Uniform("p_a", 0, 1)
        p_b = pm.Uniform("p_b", 0, 1)
        delta = pm.Deterministic("delta", p_a - p_b)  # noqa: F841
        obs_a = pm.Bernoulli("obs_a", p_a, observed=occurences_a)  # noqa: F841
        obs_b = pm.Bernoulli("obs_b", p_b, observed=occurences_b)  # noqa: F841
        step = pm.Metropolis()
        trace = pm.sample(20000, step=step)
        burned_trace = trace[1000:]

    return burned_trace, model


def get_color(i, n, name="hsv"):
    return plt.cm.get_cmap(name, n)(i)


def plot_histogram_single(ax, p_true, sample, value_name="", color="green"):
    n_colors = 12
    color = get_color(int(np.random.rand() * n_colors), n_colors)
    ax.set_title(f"histogram of {value_name}")
    ax.vlines(
        p_true,
        0,
        90,
        linestyle="--",
        label=f"true {value_name} (unknown)",
        colors=[color],
        alpha=0.7,
    )
    ax.hist(
        sample,
        bins=25,
        density=True,
        label=f"{value_name} dist.",
        alpha=0.7,
        color=color,
    )


def plot_histogram_overlap(ax, p_a_true, p_b_true, burned_trace):
    plot_histogram_single(
        ax, p_a_true, burned_trace["p_a"], value_name="$p_a$"
    )
    plot_histogram_single(
        ax, p_b_true, burned_trace["p_b"], value_name="$p_b$"
    )
    ax.legend()
    ax.grid()


def plot_histogram(p_a_true, p_b_true, trace):
    fig, axes = plt.subplots(5, 1, figsize=(12, 8))
    axes = axes.flatten()

    plot_histogram_overlap(axes[0], p_a_true, p_b_true, trace)
    plot_histogram_single(axes[1], p_a_true, trace["p_a"], value_name="$p_a$")
    plot_histogram_single(axes[2], p_b_true, trace["p_b"], value_name="$p_b$")
    plot_histogram_single(
        axes[3],
        p_b_true - p_a_true,
        trace["p_b"] - trace["p_a"],
        value_name="$p_b - p_a$",
    )
    plot_histogram_single(
        axes[4],
        (p_b_true - p_a_true) / p_a_true,
        (trace["p_b"] - trace["p_a"]) / trace["p_a"],
        value_name="$(p_b - p_a) / p_a$",
    )

    for ax in axes:
        ax.legend()
        ax.grid()

    fig.suptitle("$p_A$ と $p_B$ の事後分布と真の値")
    return fig


@click.command()
@click.option(
    "--model_output_filepath",
    type=click.Path(),
    default="models/chapter02/two_bernoulli.pickle",
)
@click.option(
    "--theta_filepath",
    type=click.Path(),
    default="models/chapter02/two_bernoulli_theta.npz",
)
@click.option("--p_a_true", type=float, default="0.05")
@click.option("--p_b_true", type=float, default="0.04")
@click.option("--n_a", type=int, default=1500)
@click.option("--n_b", type=int, default=750)
def main(**kwargs):
    # 初期値を設定
    p_a_true = kwargs["p_a_true"]
    p_b_true = kwargs["p_b_true"]
    n_a = kwargs["n_a"]
    n_b = kwargs["n_b"]

    # 観測データを生成
    occurences_a = stats.bernoulli.rvs(p_a_true, size=n_a)
    occurences_b = stats.bernoulli.rvs(p_b_true, size=n_b)

    # sampling
    trace, model = sampling(occurences_a, occurences_b)

    # save model, trace, theta, observed
    save_trace_and_model(trace, model, kwargs["model_output_filepath"])
    np.savez(
        kwargs["theta_filepath"],
        p_a_true=p_a_true,
        p_b_true=p_b_true,
        n_a=n_a,
        n_b=n_b,
        occurences_a=occurences_a,
        occurences_b=occurences_b,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
