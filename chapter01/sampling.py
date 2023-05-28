import logging
import pickle

import click
import numpy as np
import pandas as pd
import pymc3 as pm


def load_dataset(filepath: str):
    """
    データを読み込む
    """
    data = pd.read_csv(filepath)
    return data.values.T[0]


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
        fig = pm.plot_trace(trace)
    return trace, model, fig


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_trace_filepath", type=click.Path())
@click.argument("output_model_filepath", type=click.Path())
@click.option("--samples", type=int, default=40000)
@click.option("--sample_tune", type=int, default=10000)
@click.option("--chains", type=int, default=3)
def main(**kwargs):
    """
    メイン処理
    """

    # init logger
    logger = logging.getLogger(__name__)
    logger.info("process start")
    logger.info(kwargs)

    # load data
    data = load_dataset(kwargs["input_filepath"])
    logger.info(f"data: {data}")

    # modeling and sampling
    trace, model, fig = modeling(
        data,
        samples=kwargs["samples"],
        sample_tune=kwargs["sample_tune"],
        chains=kwargs["chains"],
    )
    fig.savefig("reports/figures/traceplot.png")

    # save trace and model
    with open(kwargs["output_trace_filepath"], "wb") as fo:
        pickle.dump(trace, fo)
    with open(kwargs["output_model_filepath"], "wb") as fo:
        pickle.dump(model, fo)

    # cleanup
    logger.info("process complete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
