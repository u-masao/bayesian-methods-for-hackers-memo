import logging
import os
from pathlib import Path

import click
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm


def load_dataset(filepath: str):
    """
    データを読み込む
    """
    data = pd.read_csv(filepath)
    return data.values.T[0]


def savefig(fig, save_path_string: str):
    """
    figure を保存する
    """
    save_path = Path(save_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)


def plot_observed(data):
    """
    観測値をプロットする
    """
    data_length = len(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(np.arange(data_length), data, color="#348ABD")
    ax.set_xlabel("経過日数 [days]")
    ax.set_ylabel("受信メッセージ数 [件]")
    ax.set_xlim(0, data_length)
    ax.grid()
    fig.suptitle("日別のメッセージ受信数の推移")
    return fig


def modeling(observed, samples=40000, sample_tune=10000):
    """
    モデリングとサンプリングを実行する
    """
    logger = logging.getLogger(__name__)

    alpha = 1.0 / observed.mean()
    with pm.Model() as model:  # noqa: F841
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        tau = pm.DiscreteUniform("tau", lower=0, upper=len(observed))
        index = np.arange(len(observed))
        lambda_ = pm.math.switch(tau > index, lambda_1, lambda_2)
        observation = pm.Poisson(  # noqa: F841
            "observation", lambda_, observed=observed
        )
        logger.info("start sampling")
        trace = pm.sample(samples, tune=sample_tune)
        logger.info("end sampling")
    return trace


def plot_trace(trace, data_length):
    """
    trace をプロットする
    """
    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()

    # plot lambda_1
    axes[0].hist(
        trace["lambda_1"], bins=30, alpha=0.8, label="lambda_1", density=True
    )
    axes[0].set_xlim([15, 30])

    # plot lambda_2
    axes[1].hist(
        trace["lambda_2"], bins=30, alpha=0.8, label="lambda_2", density=True
    )
    axes[1].set_xlim([15, 30])

    # plot tau
    w = 1.0 / trace["tau"].shape[0] * np.ones_like(trace["tau"])
    axes[2].hist(
        trace["tau"],
        bins=data_length,
        alpha=0.8,
        label="tau",
        weights=w,
        rwidth=2.0,
    )
    for ax in axes:
        ax.legend()
        ax.grid()
    return fig


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.option("--samples", type=int, default=40000)
@click.option("--sample_tune", type=int, default=10000)
def main(**kwargs):
    """
    メイン処理
    """
    logger = logging.getLogger(__name__)

    # load data
    data = load_dataset(kwargs["input_filepath"])
    logger.info(f"data: {data}")

    # plot data
    savefig(plot_observed(data), "reports/figures/observed.png")

    # modeling and sampling
    trace = modeling(data)

    logger.info(f'trace lambda_1.shape: {trace["lambda_1"].shape}')
    logger.info(f'trace lambda_2.shape: {trace["lambda_2"].shape}')
    logger.info(f'trace tau.shape: {trace["tau"].shape}')

    # plot trace
    savefig(plot_trace(trace, len(data)), "reports/figures/trace.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
