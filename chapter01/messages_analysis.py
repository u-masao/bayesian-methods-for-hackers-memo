import os
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(url: str):
    data = pd.read_csv(url)
    return data.values.T[0]


def savefig(fig, save_path_string: str):
    save_path = Path(save_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)


def plot_observed(
    data,
):
    data_length = len(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(np.arange(data_length), data, color="#348ABD")
    ax.set_xlabel("経過日数 [days]")
    ax.set_ylabel("受信メッセージ数 [件]")
    ax.set_xlim(0, data_length)
    ax.grid()
    fig.suptitle("日別のメッセージ受信数の推移")
    return fig


def main(**kwargs):
    data = load_dataset("https://git.io/vXTVC")
    print(data)
    savefig(plot_observed(data), "reports/figures/observed.png")


if __name__ == "__main__":
    main()
