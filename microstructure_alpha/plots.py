import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_check_features_distr(
    df,
    features,
    bins=100,
    n_cols=4,
    show=True,
    save_path=None,
    axes=None,
):

    n = len(features)
    n_rows = math.ceil(n / n_cols)

    if axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        created_fig = True
    else:
        axes = axes.flatten()
        fig = axes[0].figure
        created_fig = False

    for i, col in enumerate(features):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if created_fig:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, axes


def plot_fold_aucs(fold_aucs, title="AUC per Fold", show=True, save_path=None):

    plt.figure(figsize=(4, 2))
    plt.plot(fold_aucs, marker="o", label="AUC")

    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


def plot_rolling_calibration(preds, y_true, window=500, show=True, save_path=None):

    pred_smooth = pd.Series(preds).rolling(window).mean()
    target_smooth = pd.Series(y_true).rolling(window).mean()

    plt.figure(figsize=(16, 3))
    plt.plot(pred_smooth, label="Predicted prob (smoothed)")
    plt.plot(target_smooth, label="True (rolling)", alpha=0.8)
    plt.xlim(0, len(y_true))
    plt.legend()
    plt.title(f"Rolling Calibration (window={window})")

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def plot_quantile_accuracy(
    all_results,
    y_true,
    quantiles=(0.8, 0.9, 0.95),
):

    y_true = np.asarray(y_true)

    model_names = list(all_results.keys())

    x = np.arange(len(model_names))

    plt.figure(figsize=(10, 6))

    for q in quantiles:

        accs = []

        for name in model_names:

            fold_oof = all_results[name]["fold_oof"]

            preds = np.nansum(fold_oof, axis=0)
            threshold = np.quantile(preds, q)
            signal = preds > threshold

            acc = y_true[signal].mean() if signal.sum() > 0 else np.nan
            accs.append(acc)

        plt.scatter(x, accs, label=f"q={q}", s=80)

    plt.xticks(x, model_names, rotation=45)
    plt.ylabel("Accuracy (conditional)")
    plt.title("Model Comparison at Different Quantiles")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_calibration(
    all_results,
    y,
    window=500,
    figsize_per_model=3,
    show=True,
    save_path=None,
):

    y_true = np.asarray(y)
    target_smooth = pd.Series(y_true).rolling(window).mean()

    n_models = len(all_results)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(12, figsize_per_model * n_models), sharex=True
    )

    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):

        preds = np.nansum(res["fold_oof"], axis=0)
        pred_smooth = pd.Series(preds).rolling(window).mean()

        ax.plot(pred_smooth, label=name)
        ax.plot(target_smooth, color="black", linewidth=2, label="true")

        ax.set_title(name)
        ax.grid(alpha=0.2)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
