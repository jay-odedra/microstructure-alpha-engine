import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from microstructure_alpha.utils.stats import block_bootstrap_sem


def plot_feature_overview(
    df,
    cols,
    figsize=(16, 6),
    lags=100,
    show=True,
    save_path=None,
):

    for col in cols:

        s = df[col].dropna()

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 4)

        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, :]),
            fig.add_subplot(gs[0, 3]),
        ]

        s.plot(ax=ax[0])
        ax[0].set_title(f"{col} over time")

        q_low, q_high = s.quantile([0.01, 0.99])
        s.clip(q_low, q_high).hist(ax=ax[1], bins=100)
        ax[1].set_title(f"{col} (1–99% clipped)")

        s.plot.box(ax=ax[2])
        ax[2].set_title(f"{col} boxplot")

        acf_vals = acf(s, nlags=lags)
        acf_sum = np.sum(acf_vals[1:])

        plot_acf(s, ax=ax[3], lags=lags)
        ax[3].set_title(f"{col} autocorrelation")
        ax[3].plot([], [], label=f"sum={acf_sum:.4f}")
        ax[3].legend()

        stats = f"""
        mean: {s.mean():.2e}
        std:  {s.std():.2e}
        min:  {s.min():.2e}
        25%:  {s.quantile(0.25):.2e}
        50%:  {s.quantile(0.50):.2e}
        75%:  {s.quantile(0.75):.2e}
        max:  {s.max():.2e}

        top:
        {s.round(8).value_counts().head(5)}
        """

        ax[4].axis("off")
        ax[4].text(0, 1, stats, va="top", family="monospace")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(f"{save_path}/{col}.png", dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


def conditional_expectation_plotter(
    df,
    features,
    target,
    block_size=5,
    n_boot=100,
    use_uncertainty=True,
    show=True,
    save_path=None,
):
    num_plots = len(features)
    cols = min(3, num_plots)
    rows = math.ceil(num_plots / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    ax = np.atleast_1d(ax).flatten()

    for i, feature in enumerate(features):

        data = df[[feature, target]]
        bins = pd.qcut(data[feature], 10, duplicates="drop")
        grouped = data.groupby(bins)

        means = grouped[target].mean()
        bin_means = grouped[feature].mean()

        if use_uncertainty:
            yerr = grouped[target].apply(
                lambda x: block_bootstrap_sem(x, block_size, n_boot)
            )
        else:
            yerr = None
        x = np.arange(len(means))
        ax[i].errorbar(
            x,
            means,
            yerr=yerr,
            capsize=3 if use_uncertainty else 0,
            marker="o",
        )

        ax[i].set_title(feature)
        ax[i].set_xlabel(feature)
        ax[i].set_ylabel(target)

    for j in range(len(features), len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
