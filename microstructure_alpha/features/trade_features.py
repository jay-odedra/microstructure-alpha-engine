from microstructure_alpha.utils.constants import EPS


def trade_base_features(trade_df):

    df = trade_df.copy()

    df["buy_trade"] = (df["isBuyerMaker"] == False).astype(int)
    df["sell_trade"] = (df["isBuyerMaker"] == True).astype(int)

    df["buy_qty"] = df["qty"].where(df["isBuyerMaker"] == False, 0)
    df["sell_qty"] = df["qty"].where(df["isBuyerMaker"] == True, 0)

    df["price_qty"] = df["price"] * df["qty"]

    agg = df.groupby("timestamp").agg(
        trade_count=("qty", "count"),
        buy_count=("buy_trade", "sum"),
        sell_count=("sell_trade", "sum"),
        total_trade_volume=("qty", "sum"),
        buy_volume=("buy_qty", "sum"),
        sell_volume=("sell_qty", "sum"),
        avg_trade_size=("qty", "mean"),
        max_trade_size=("qty", "max"),
        min_trade_size=("qty", "min"),
        std_trade_size=("qty", "std"),
        vwap_num=("price_qty", "sum"),
    )

    agg["vwap"] = agg["vwap_num"] / (agg["total_trade_volume"] + EPS)
    agg["max_over_average"] = agg["max_trade_size"] / agg["avg_trade_size"]
    agg = agg.drop(columns="vwap_num")

    return agg.reset_index()


def trade_pressure_features(df):
    df["trade_volume_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / (
        df["buy_volume"] + df["sell_volume"] + EPS
    )

    return df


def trade_change_features(df):
    df["trade_count_imbalance"] = (df["buy_count"] - df["sell_count"]) / (
        df["buy_count"] + df["sell_count"] + 1e-9
    )
    df["trade_volume_change"] = df["total_trade_volume"].diff()

    df["trade_count_change"] = df["trade_count"].diff()

    return df


def trade_lag_features(df):

    df["lag_trade_volume_imbalance_1"] = df["trade_volume_imbalance"].shift(1)
    df["lag_trade_volume_imbalance_2"] = df["trade_volume_imbalance"].shift(2)
    df["lag_trade_volume_imbalance_3"] = df["trade_volume_imbalance"].shift(3)
    df["lag_trade_volume_imbalance_5"] = df["trade_volume_imbalance"].shift(5)

    return df


def build_trade_feature_pipeline(df):
    df = trade_base_features(df)

    df = trade_pressure_features(df)

    df = trade_change_features(df)

    df = trade_lag_features(df)
    return df
