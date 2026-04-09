import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from microstructure_alpha.utils.constants import EPS
from microstructure_alpha.utils.stats import block_bootstrap_sem


def feature_scoring(df, feature, target, block_size=10):

    data = df[[feature, target]].copy()
    bins = pd.qcut(data[feature], 10, duplicates="drop")

    grouped = data.groupby(bins)

    means = grouped[target].mean().values
    count = grouped[target].count().values
    sems = (
        grouped[target].apply(lambda x: block_bootstrap_sem(x, block_size, 100)).values
    )
    bin_means = grouped[feature].mean().values.reshape(-1, 1)

    eps = EPS

    model = LinearRegression().fit(
        bin_means,
        means,
        sample_weight=1 / (sems**2 + eps),
    )

    y_pred = model.predict(bin_means)
    residuals = means - y_pred
    non_lin_score = np.mean((residuals / (sems + eps)) ** 2)

    weight_bin = count / count.sum()

    signal_strength = (np.abs(means) * weight_bin).sum()
    s_over_noise_weighted = ((np.abs(means) / (sems + eps)) * weight_bin).sum()
    stability = ((np.abs(means) > sems) * weight_bin).sum()

    return {
        "feature": feature,
        "signal_strength": signal_strength,
        "s_over_noise_weighted": s_over_noise_weighted,
        "nonlinearity": non_lin_score,
        "stability": stability,
    }
