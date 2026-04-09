import numpy as np


def signed_log1p(x):
    x = x.astype(float).copy()
    x = np.sign(x) * np.log1p(np.abs(x))

    return x


def apply_transform(df, features, func, suffix, use_values=True):
    df = df.copy()

    for col in features:
        x = df[col].values if use_values else df[col]
        df[f"{col}_{suffix}"] = func(x)

    return df