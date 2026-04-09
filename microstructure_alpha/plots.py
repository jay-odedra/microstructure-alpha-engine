import math
from pathlib import Path

import matplotlib.pyplot as plt


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
