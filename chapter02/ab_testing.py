import os
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy.stats as stats


def savefig(fig, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    fig.savefig(path)


def main():
    model = pm.Model()
    with model:
        p = pm.Uniform("p", lower=0, upper=1)

    p_true = 0.05
    N = 1500
    occurrences = stats.bernoulli.rvs(p_true, size=N)
    print("occurrence:", occurrences)
    print("sum:", np.sum(occurrences))
    print("mean:", np.mean(occurrences))
    print("mean - p_true:", np.mean(occurrences) - p_true)
    print("(mean - p_true)/ p_true:", (np.mean(occurrences) - p_true) / p_true)

    with model:
        obs = pm.Bernoulli("obs", p, observed=occurrences)  # noqa: F841
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        burned_trace = trace[1000:]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    ax.hist(burned_trace["p"], bins=25, histtpye="stepfilled", density=True)
    fig.suptitle("$p_A$ の事後分布と真の $p_A$")
    savefig(fig, "reports/figures/ch02/bernoulli.png")


if __name__ == "__main__":
    main()
