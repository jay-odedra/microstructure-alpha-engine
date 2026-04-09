import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def block_bootstrap_sem(x, block_size=5, n_boot=100):
    x = np.asarray(x)
    n = len(x)
    means = []

    for _ in range(n_boot):
        sample = []

        while len(sample) < n:
            i = np.random.randint(0, n - block_size)
            sample.extend(x[i : i + block_size])

        sample = np.array(sample[:n])
        means.append(sample.mean())

    return np.std(means)


def compute_signal_autocorr(
    df,
    feature,
    target,
    n_bins=20,
    nlags=100,
    plot=True,
    ax=None,
):
    data = df[[feature, target]].dropna()

    data["bin"] = pd.qcut(data[feature], n_bins, duplicates="drop")

    data["y_hat"] = data.groupby("bin")[target].transform("mean")

    y_hat = data["y_hat"].values

    acf_vals = acf(y_hat, nlags=nlags)

    tau_int = 1 + 2 * np.sum(acf_vals[1:][acf_vals[1:] > 0])

    below = np.where(acf_vals[1:] < 0.1)[0]
    tau_cutoff = below[0] + 1 if len(below) > 0 else nlags

    # plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(acf_vals, marker="o")
        ax.axhline(0, linestyle="--")
        ax.axvline(tau_cutoff, color="red", linestyle="--")
        ax.set_title(f"Signal ACF: {feature}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

    return tau_int, tau_cutoff
