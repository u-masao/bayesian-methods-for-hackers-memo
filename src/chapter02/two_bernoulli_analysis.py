import logging
from pathlib import Path

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import (
    calc_credible_intervals,
    load_trace_and_model,
    plot_trace,
    savefig,
)


def log_metrics(occurences, p_true, label):
    logger = logging.getLogger(__name__)
    metrics = {
        "occurrence": occurences,
        "len": len(occurences),
        "sum": np.sum(occurences),
        "mean": np.mean(occurences),
        "mean - p_true:": np.mean(occurences) - p_true,
        "(mean - p_true) / p_true:": (np.mean(occurences) - p_true) / p_true,
    }

    logger.info(f"{label} 観測値と真の値: \n{metrics}")


def plot_histogram_single(
    ax, p_true, sample, value_name="", color=None, hdi_prob=0.95
):
    # 描画色を指定
    if color is None:
        color = plt.get_cmap("Dark2")(int(np.random.rand() * 10))

    # 確信区間を計算
    ci_low, ci_high = calc_credible_intervals(sample, hdi_prob=hdi_prob)

    # タイトルを指定
    ax.set_title(f"{value_name} の分布")

    # plot
    n, _, _ = ax.hist(
        sample,
        bins=25,
        density=True,
        label=f"{value_name} の分布",
        alpha=0.4,
        color=color,
    )
    ax.vlines(
        p_true,
        0,
        np.max(n) * 1.2,
        linestyle="--",
        label=f"{value_name} の真の値",
        colors=[color],
        alpha=0.5,
    )
    ax.plot(
        [ci_low, ci_high],
        [np.max(n) * (0.1 + 0.2 * np.random.rand())] * 2,
        linestyle="-",
        label=f"{hdi_prob * 100:0.0f} 確信区間 {value_name}",
        color=color,
        alpha=0.9,
        marker="x",
    )


def plot_histogram_overlap(ax, p_a_true, p_b_true, burned_trace):
    plot_histogram_single(
        ax,
        p_a_true,
        burned_trace["p_a"],
        value_name="$p_a$",
        color=plt.get_cmap("Dark2")(0),
    )
    plot_histogram_single(
        ax,
        p_b_true,
        burned_trace["p_b"],
        value_name="$p_b$",
        color=plt.get_cmap("Dark2")(1),
    )
    ax.set_title("$p_a$ と $p_b$ のヒストグラム")


def plot_histogram(p_a_true, p_b_true, trace):
    """plot histogram"""
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
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

    # 軸を一致
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])

    for ax in axes:
        ax.legend()
        ax.grid()

    fig.suptitle("$p_A$ と $p_B$ の事後分布と真の値")
    fig.tight_layout()
    return fig


def load_theta(filepath):
    """load true parameters"""
    theta = np.load(filepath)
    p_a_true = theta["p_a_true"]
    p_b_true = theta["p_b_true"]
    n_a = theta["n_a"]
    n_b = theta["n_b"]
    occurences_a = theta["occurences_a"]
    occurences_b = theta["occurences_b"]
    return p_a_true, p_b_true, n_a, n_b, occurences_a, occurences_b


def calc_ci(p_a, p_b, hdi_prob=0.95):
    # init log
    logger = logging.getLogger(__name__)

    p_diff = p_b - p_a
    p_ratio = p_diff / p_a

    # 確信区間を計算
    p_a_ci_low, p_a_ci_high = calc_credible_intervals(p_a, hdi_prob=hdi_prob)
    p_b_ci_low, p_b_ci_high = calc_credible_intervals(p_b, hdi_prob=hdi_prob)
    p_diff_ci_low, p_diff_ci_high = calc_credible_intervals(
        p_diff, hdi_prob=hdi_prob
    )
    p_ratio_ci_low, p_ratio_ci_high = calc_credible_intervals(
        p_ratio, hdi_prob=hdi_prob
    )
    ci = {
        "p_a": {"low": p_a_ci_low, "high": p_a_ci_high},
        "p_b": {"low": p_b_ci_low, "high": p_b_ci_high},
        "p_diff": {"low": p_diff_ci_low, "high": p_diff_ci_high},
        "p_ratio": {"low": p_ratio_ci_low, "high": p_ratio_ci_high},
    }
    logger.info(f"credible intarvals: {ci}")
    return ci


def calc_prob_dist(p_a, p_b, ci):
    values = np.linspace(ci["p_diff"]["low"], ci["p_diff"]["high"], 100)
    prob = [(p_a < p_b + x).mean() for x in values]
    prob2 = [(p_a - p_b < x).mean() for x in values]

    result_df = pd.DataFrame(
        [values, prob, prob2], columns=["diff", "prob", "prob2"]
    )
    return result_df


def calc_prob_for_dicision(
    trace, model, p_a_true, p_b_true, occurences_a, occurences_b, hdi_prob=0.95
):
    p_a = trace["p_a"]
    p_b = trace["p_b"]
    # p_diff = p_b - p_a
    # p_ratio = p_diff / p_a
    ci = calc_ci(p_a, p_b, hdi_prob=hdi_prob)

    # 差と確率
    prob_diff_df = calc_prob_dist(p_a, p_b, ci)

    return ci, prob_diff_df


def calc_compare_trueth_and_prob(
    trace, model, p_a_true, p_b_true, occurences_a, occurences_b
):
    pass


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("theta_filepath", type=click.Path(exists=True))
@click.option(
    "--figure_dir", type=click.Path(), default="reports/figures/chapter02/"
)
def main(**kwargs):
    # load model, trace, theta
    trace, model = load_trace_and_model(kwargs["model_filepath"])
    p_a_true, p_b_true, n_a, n_b, occurences_a, occurences_b = load_theta(
        kwargs["theta_filepath"]
    )

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

    # ログ出力
    log_metrics(occurences_a, p_a_true, "a")
    log_metrics(occurences_b, p_b_true, "b")

    # 意思決定に利用する確率などの計算

    ci, prob_diff_df = calc_prob_for_dicision(
        trace, model, p_a_true, p_b_true, occurences_a, occurences_b
    )
    prob_diff_df.to_csv("data/processed/prob_diff.csv")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
