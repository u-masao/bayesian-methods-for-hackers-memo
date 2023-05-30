import os
import pickle
from pathlib import Path


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
