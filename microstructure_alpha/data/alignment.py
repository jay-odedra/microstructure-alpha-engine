import pandas as pd


def explode_lob(df, side: str, level: int = 10):

    df = df.copy()
    df[side] = df[side].str[:level]
    # fix latency
    df["timestamp"] = (df["timestamp"] - df["latency"]).astype("int64")

    out = df[["timestamp", "lastUpdateId", side]].explode(side)

    out[f"lob_{side}_price"], out[f"lob_{side}_volume"] = zip(*out[side])
    out[f"lob_{side}_price"] = out[f"lob_{side}_price"].astype(float)
    out[f"lob_{side}_volume"] = out[f"lob_{side}_volume"].astype(float)

    out["level"] = out.groupby("timestamp").cumcount() + 1
    out = out.drop(columns=side)
    return out


def pivot_lob_side_levels(df, side: str):
    prices = df.pivot(index="timestamp", columns="level", values=f"lob_{side}_price")
    volume = df.pivot(index="timestamp", columns="level", values=f"lob_{side}_volume")

    prices.columns = [f"lob_{side}_price_{i}" for i in prices.columns]
    volume.columns = [f"lob_{side}_volume_{i}" for i in volume.columns]

    prices.reset_index()
    volume.reset_index()

    return pd.concat([prices, volume], axis=1)


def process_lob(lob_df):
    bids_df_10 = explode_lob(lob_df, "bids")
    asks_df_10 = explode_lob(lob_df, "asks")
    combined_lob_df = pd.concat(
        [
            pivot_lob_side_levels(bids_df_10, "bids"),
            pivot_lob_side_levels(asks_df_10, "asks"),
        ],
        axis=1,
    )
    combined_lob_df.reset_index(inplace=True)
    return combined_lob_df


def process_trade(trade_df, combined_lob_df):

    trade_df = trade_df.astype(
        {
            "price": float,
            "qty": float,
            "quoteQty": float,
            "time": "int64",
            "isBuyerMaker": bool,
            "isBestMatch": bool,
        }
    )

    trade_df = trade_df.sort_values("time")
    combined_lob_df = combined_lob_df.sort_values("timestamp")

    trade_df = pd.merge_asof(
        trade_df,
        combined_lob_df[["timestamp"]],
        left_on="time",
        right_on="timestamp",
        direction="backward",
        tolerance=1000,
    )

    trade_df = trade_df.dropna(subset=["timestamp"])

    assert (trade_df["time"] < trade_df["timestamp"]).sum() == 0

    return trade_df
