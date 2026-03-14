import numpy as np
import pandas as pd
from microstructure_alpha.utils.constants import EPS


def create_lob_price_features(df):
    df["mid_price"] = (df["lob_bids_price_1"] + df["lob_asks_price_1"]) / 2
    df["spread"] = df["lob_asks_price_1"] - df["lob_bids_price_1"]
    df["rel_spread"] = df["spread"] / (df["mid_price"] + EPS)
    return df


def lob_volume_features(df):
    df["liquidity"] = df["lob_bids_volume_1"] + df["lob_asks_volume_1"]
    df["total_bid_volume_10"] = df[[f"lob_bids_volume_{i}" for i in range(1, 11)]].sum(
        axis=1
    )
    df["total_ask_volume_10"] = df[[f"lob_asks_volume_{i}" for i in range(1, 11)]].sum(
        axis=1
    )
    df["total_book_volume"] = df["total_ask_volume_10"] + df["total_bid_volume_10"]
    df["max_bid_ask_vol_ratio"] = df["lob_bids_volume_1"] / (
        df["lob_asks_volume_1"] + EPS
    )
    return df


def lob_pressure_features(df):

    df["imbalance_1"] = (df["lob_bids_volume_1"] - df["lob_asks_volume_1"]) / (
        df["lob_bids_volume_1"] + df["lob_asks_volume_1"] + EPS
    )

    bid_vol5 = df[[f"lob_bids_volume_{i}" for i in range(1, 6)]].sum(axis=1)
    ask_vol5 = df[[f"lob_asks_volume_{i}" for i in range(1, 6)]].sum(axis=1)

    df["imbalance_5"] = (bid_vol5 - ask_vol5) / (bid_vol5 + ask_vol5 + EPS)

    bid10 = df[[f"lob_bids_volume_{i}" for i in range(1, 11)]].sum(axis=1)
    ask10 = df[[f"lob_asks_volume_{i}" for i in range(1, 11)]].sum(axis=1)

    df["imbalance_10"] = (bid10 - ask10) / (bid10 + ask10 + EPS)

    df["microprice"] = (
        df["lob_bids_volume_1"] * df["lob_asks_price_1"]
        + df["lob_asks_volume_1"] * df["lob_bids_price_1"]
    ) / (df["liquidity"] + EPS)

    df["microprice_change"] = df["microprice"].diff()
    df["mid_minus_micro"] = df["mid_price"] - df["microprice"]

    numerator10 = sum(
        df[f"lob_bids_volume_{i}"] * df[f"lob_asks_price_{i}"]
        + df[f"lob_asks_volume_{i}"] * df[f"lob_bids_price_{i}"]
        for i in range(1, 11)
    )

    denominator10 = sum(
        df[f"lob_bids_volume_{i}"] + df[f"lob_asks_volume_{i}"] for i in range(1, 11)
    )
    df["microprice_weighted_10"] = numerator10 / (denominator10 + EPS)
    return df


def lob_returns_and_momentum_features(df):
    df["return_1"] = df["mid_price"].pct_change(1)
    df["return_5"] = df["mid_price"].pct_change(5)

    df["log_return_1"] = np.log(df["mid_price"]).diff(1)
    df["log_return_2"] = np.log(df["mid_price"]).diff(2)
    df["log_return_3"] = np.log(df["mid_price"]).diff(3)
    df["log_return_5"] = np.log(df["mid_price"]).diff(5)
    df["log_return_20"] = np.log(df["mid_price"]).diff(20)
    # momentum based return

    df["momentum_5_log_return_1"] = df["log_return_1"].rolling(5).mean()
    df["momentum_20_log_return_1"] = df["log_return_1"].rolling(20).mean()

    return df


def lob_volatility_features(df):
    df["vol_5"] = df["log_return_1"].rolling(5).std()
    df["vol_20"] = df["log_return_1"].rolling(20).std()

    df["realized_vol_5"] = np.sqrt((df["log_return_1"] ** 2).rolling(5).sum())
    df["realized_vol_20"] = np.sqrt((df["log_return_1"] ** 2).rolling(20).sum())
    return df


def lob_target_feature(df):
    df["mid_price_change"] = df["mid_price"].shift(-1) - df["mid_price"]
    df["mid_price_change_sign"] = np.sign(df["mid_price_change"])
    return df


def build_lob_feature_pipeline(df):

    df = create_lob_price_features(df)

    df = lob_volume_features(df)

    df = lob_pressure_features(df)

    df = lob_returns_and_momentum_features(df)

    df = lob_volatility_features(df)

    df = lob_target_feature(df)

    return df
