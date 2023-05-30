import logging
from pathlib import Path

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

from src.utils import plot_trace, save_trace_and_model, savefig


def log_metrics(occurences, p_true, label):
    logger = logging.getLogger(__name__)
    metrics = {
        "occurrence": occurences,
        "sum": np.sum(occurences),
        "mean": np.mean(occurences),
        "mean - p_true:": np.mean(occurences) - p_true,
        "(mean - p_true) / p_true:": (np.mean(occurences) - p_true) / p_true,
    }

    logger.info(f"{label} metrics: \n{metrics}")


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


def plot_histogram_overlap(ax, p_a_true, p_b_true, burned_trace):
    ax.vlines(p_a_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    ax.vlines(p_b_true, 0, 90, linestyle="--", label="true $p_B$ (unknown)")
    ax.hist(
        burned_trace["p_a"],
        bins=25,
        density=True,
        label="$p_a$ dist.",
        alpha=0.5,
    )
    ax.hist(
        burned_trace["p_b"],
        bins=25,
        density=True,
        label="$p_b$ dist.",
        alpha=0.5,
    )

    ax.legend()
    ax.grid()


def plot_histogram(p_a_true, p_b_true, burned_trace):
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    plot_histogram_overlap(axes[0], p_a_true, p_b_true, burned_trace)

    fig.suptitle("$p_A$ と $p_B$ の事後分布と真の値")
    return fig


@click.command()
@click.option(
    "--figure_dir", type=click.Path(), default="reports/figures/chapter02/"
)
@click.option(
    "--model_output_filepath",
    type=click.Path(),
    default="models/chapter02/two_bernoulli.pickle",
)
def main(**kwargs):
    p_a_true = 0.05
    p_b_true = 0.04
    n_a = 1500
    n_b = 750

    occurences_a = stats.bernoulli.rvs(p_a_true, size=n_a)
    occurences_b = stats.bernoulli.rvs(p_b_true, size=n_b)

    log_metrics(occurences_a, p_a_true, "a")
    log_metrics(occurences_a, p_a_true, "a")
    log_metrics(occurences_b, p_b_true, "b")

    trace, model = sampling(occurences_a, occurences_b)
    save_trace_and_model(trace, model, kwargs["model_output_filepath"])
    savefig(
        plot_trace(trace, model),
        Path(kwargs["figure_dir"]) / "trace.png",
    )

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
