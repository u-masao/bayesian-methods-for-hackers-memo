import logging
from pathlib import Path

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats

from src.utils import save_trace_and_model, savefig


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


@click.command()
@click.option(
    "--figure_dir", type=click.Path(), default="reports/figures/chapter02/"
)
@click.option(
    "--model_output_filepath",
    type=click.Path(),
    default="models/chapter02/one_bernoulli.pickle",
)
def main(**kwargs):
    model = pm.Model()
    with model:
        p = pm.Uniform("p", lower=0, upper=1)

    p_true = 0.05
    N = 1500
    occurrences = stats.bernoulli.rvs(p_true, size=N)
    log_metrics(occurrences, p_true, "a")

    with model:
        obs = pm.Bernoulli("obs", p, observed=occurrences)  # noqa: F841
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        burned_trace = trace[1000:]

    save_trace_and_model(trace, model, kwargs["model_output_filepath"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    ax.hist(burned_trace["p"], bins=25, density=True)
    ax.legend()
    ax.grid()
    fig.suptitle("$p_A$ の事後分布と真の $p_A$")
    savefig(fig, Path(kwargs["figure_dir"]) / "bernoulli.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
