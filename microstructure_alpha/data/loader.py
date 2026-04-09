from pathlib import Path

import pandas as pd


def load_parquet_glob(path, sort_by=None):
    path = Path(path)

    files = list(path.parent.glob(path.name))
    assert len(files) > 0, f"No files found for pattern: {path}"

    df = pd.concat([pd.read_parquet(f) for f in files])

    if sort_by is not None:
        df = df.sort_values(sort_by).reset_index(drop=True)

    return df
