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


def modeling(observed, samples=40000, sample_tune=10000, chains=3):
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
        trace = pm.sample(samples, tune=sample_tune, chains=chains)
        logger.info("end sampling")
    return trace


def plot_trace(trace, data):
    """
    trace をプロットする
    """

    # calc length
    data_length = len(data)

    # 日別の期待値を計算
    expected_messages_per_day = np.zeros(data_length)
    for day in range(data_length):
        ix = day < trace["tau"]
        expected_messages_per_day[day] = (
            trace["lambda_1"][ix].sum() + trace["lambda_2"][~ix].sum()
        ) / len(trace["tau"])

    # チャートを初期化
    fig, axes = plt.subplots(4, 1, figsize=(8, 6))
    axes = axes.flatten()

    # plot lambda_1
    axes[0].hist(
        trace["lambda_1"],
        bins=30,
        alpha=0.8,
        label=r"$\lambda_1$",
        density=True,
    )
    axes[0].set_xlim([15, 30])
    axes[0].set_title(r"$\lambda_1$ の事後分布")

    # plot lambda_2
    axes[1].hist(
        trace["lambda_2"],
        bins=30,
        alpha=0.8,
        label=r"$\lambda_2$",
        density=True,
    )
    axes[1].set_xlim([15, 30])
    axes[1].set_title(r"$\lambda_2$ の事後分布")

    # plot tau
    axes[2].hist(
        trace["tau"],
        bins=data_length,
        alpha=0.8,
        label=r"$\tau$",
        weights=1.0 / trace["tau"].shape[0] * np.ones_like(trace["tau"]),
        rwidth=2.0,
    )
    axes[2].set_title(r"$\tau$ の事後分布")

    # plot tau estimated mean
    axes[3].bar(np.arange(data_length), data, label="生データ")
    axes[3].plot(
        range(data_length),
        expected_messages_per_day,
        label="受信メッセージ数の期待値",
        color="orange",
    )
    axes[3].set_title("受信メッセージ数の期待値")

    # add chart parts
    for ax in axes:
        ax.legend()
        ax.grid()

    fig.tight_layout()
    return fig


def plot_effects(trace, data):
    """
    増加量に対して発生確率をプロット
    """
    effects = (
        np.linspace(trace["lambda_1"].min(), trace["lambda_2"].max(), 1000)
        - trace["lambda_1"].mean()
    )
    proba = []
    for effect in effects:
        proba.append((trace["lambda_1"] + effect < trace["lambda_2"]).mean())
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(effects, proba, label=r"$\lambda_1$ よりも $\lambda_2$ の強度が強い確率")
    ax.set_xlabel("強度の効果量")
    ax.set_ylabel("確率")
    ax.legend()
    ax.grid()
    fig.suptitle("強度の変化が推定される確率")
    return fig


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.option("--samples", type=int, default=40000)
@click.option("--sample_tune", type=int, default=10000)
@click.option("--chains", type=int, default=3)
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
    trace = modeling(
        data,
        samples=kwargs["samples"],
        sample_tune=kwargs["sample_tune"],
        chains=kwargs["chains"],
    )

    logger.info(f'trace lambda_1.shape: {trace["lambda_1"].shape}')
    logger.info(f'trace lambda_2.shape: {trace["lambda_2"].shape}')
    logger.info(f'trace tau.shape: {trace["tau"].shape}')

    # plot trace
    savefig(plot_trace(trace, data), "reports/figures/trace.png")

    # test effects
    savefig(plot_effects(trace, data), "reports/figures/effects.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
