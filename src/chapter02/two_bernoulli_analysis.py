import logging
from pathlib import Path

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_trace_and_model, plot_trace, savefig


def get_color(i, n, name="brg"):
    return plt.cm.get_cmap(name, n)(i)


def plot_histogram_single(ax, p_true, sample, value_name="", color="green"):
    n_colors = 12
    color = get_color(int(np.random.rand() * n_colors), n_colors)
    ax.set_title(f"histogram of {value_name}")
    n, _, _ = ax.hist(
        sample,
        bins=25,
        density=True,
        label=f"{value_name} dist.",
        alpha=0.5,
        color=color,
    )
    ax.vlines(
        p_true,
        0,
        np.max(n) * 1.2,
        linestyle="--",
        label=f"true {value_name} (unknown)",
        colors=[color],
        alpha=0.9,
    )


def plot_histogram_overlap(ax, p_a_true, p_b_true, burned_trace):
    plot_histogram_single(
        ax, p_a_true, burned_trace["p_a"], value_name="$p_a$"
    )
    plot_histogram_single(
        ax, p_b_true, burned_trace["p_b"], value_name="$p_b$"
    )


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
    fig.tight_layout()
    return fig


def load_theta(filepath):
    theta = np.load(filepath)
    p_a_true = theta["p_a_true"]
    p_b_true = theta["p_b_true"]
    n_a = theta["n_a"]
    n_b = theta["n_b"]
    return p_a_true, p_b_true, n_a, n_b


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("theta_filepath", type=click.Path(exists=True))
@click.option(
    "--figure_dir", type=click.Path(), default="reports/figures/chapter02/"
)
def main(**kwargs):
    # load model, trace, theta
    trace, model = load_trace_and_model(kwargs["model_filepath"])
    p_a_true, p_b_true, n_a, n_b = load_theta(kwargs["theta_filepath"])

    # plot trace
    savefig(
        plot_trace(trace, model),
        Path(kwargs["figure_dir"]) / "trace.png",
    )

    # plot histogram
    savefig(
        plot_histogram(
            p_a_true,
            p_b_true,
            trace,
        ),
        Path(kwargs["figure_dir"]) / "bernoulli.png",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
