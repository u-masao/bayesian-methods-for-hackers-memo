import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


def get_color(i: int, n: int = 20, name: str = "brg"):
    return plt.cm.get_cmap(name, n)(i)


def calc_credible_intervals(data, hdi_prob=0.95):
    assert hdi_prob <= 1.0
    assert hdi_prob >= 0.0
    low_prob = (0.5 - hdi_prob / 2.0) * 100.0
    high_prob = (0.5 + hdi_prob / 2.0) * 100.0

    return np.percentile(data, q=[low_prob, high_prob])


def plot_trace(trace, model):
    # save trace plot
    with model:
        axes = pm.plot_trace(trace, compact=False, combined=False)
        for ax in axes.flatten():
            ax.grid()
        fig = axes.ravel()[0].figure
        fig.tight_layout()
        fig.suptitle("trace plot")
    return fig


def savefig(fig, save_path_string: str) -> None:
    """
    figure を保存する
    """
    save_path = Path(save_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)


def load_trace_and_model(model_path: str):
    # load trace and model
    with open(model_path, "rb") as fo:
        trace, model = pickle.load(fo)
    return trace, model


def save_trace_and_model(trace, model, model_path_string: str):
    # save trace and model
    save_path = Path(model_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as fo:
        pickle.dump((trace, model), fo)
