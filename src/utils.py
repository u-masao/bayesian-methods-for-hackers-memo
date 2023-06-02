import os
import pickle
from pathlib import Path

import numpy as np
import pymc3 as pm


def calc_credible_intervals(data, hdi_prob=0.95):
    assert hdi_prob <= 1.0
    assert hdi_prob >= 0.0
    prob_low = (0.5 - hdi_prob / 2.0) * 100.0
    prob_high = (0.5 + hdi_prob / 2.0) * 100.0
    ci_low, ci_high = np.percentile(data, q=[prob_low, prob_high])
    return ci_low, ci_high


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
