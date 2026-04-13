lob_features = [
    "lob_bids_price_1",
    "lob_bids_price_2",
    "lob_bids_price_3",
    "lob_bids_price_4",
    "lob_bids_price_5",
    "lob_bids_price_6",
    "lob_bids_price_7",
    "lob_bids_price_8",
    "lob_bids_price_9",
    "lob_bids_price_10",
    "lob_bids_volume_1",
    "lob_bids_volume_2",
    "lob_bids_volume_3",
    "lob_bids_volume_4",
    "lob_bids_volume_5",
    "lob_bids_volume_6",
    "lob_bids_volume_7",
    "lob_bids_volume_8",
    "lob_bids_volume_9",
    "lob_bids_volume_10",
    "lob_asks_price_1",
    "lob_asks_price_2",
    "lob_asks_price_3",
    "lob_asks_price_4",
    "lob_asks_price_5",
    "lob_asks_price_6",
    "lob_asks_price_7",
    "lob_asks_price_8",
    "lob_asks_price_9",
    "lob_asks_price_10",
    "lob_asks_volume_1",
    "lob_asks_volume_2",
    "lob_asks_volume_3",
    "lob_asks_volume_4",
    "lob_asks_volume_5",
    "lob_asks_volume_6",
    "lob_asks_volume_7",
    "lob_asks_volume_8",
    "lob_asks_volume_9",
    "lob_asks_volume_10",
    "lob_depth_ratio_2",
    "lob_depth_ratio_3",
    "lob_depth_ratio_4",
    "lob_depth_ratio_5",
    "lob_depth_ratio_6",
    "lob_depth_ratio_7",
    "lob_depth_ratio_8",
    "lob_depth_ratio_9",
    "lob_depth_ratio_10",
]

spread_features = ["rel_spread", "spread"]
imbalance_features = [
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "imbalance_depth_1",
    "imbalance_depth_2",
    "imbalance_depth_3",
    "imbalance_depth_4",
    "imbalance_depth_5",
    "imbalance_depth_6",
    "imbalance_depth_7",
    "imbalance_depth_8",
    "imbalance_depth_9",
    "imbalance_depth_10",
]
liquidity_features = [
    "liquidity",
    "total_bid_volume_10",
    "total_ask_volume_10",
    "total_book_volume",
    "max_bid_ask_vol_ratio",
]
microprice_features = [
    "microprice",
    "microprice_change",
    "mid_minus_micro",
    "microprice_weighted_10",
]
return_features = [
    "return_1",
    "return_5",
    "log_return_1",
    "log_return_2",
    "log_return_3",
    "log_return_5",
    "log_return_20",
    "mid_price_change_1",
    "mid_price_change_5",
    "mid_price_change_20",
]
momentum_features = [
    "momentum_5_log_return_1",
    "momentum_20_log_return_1",
]
volatility_features = [
    "vol_5",
    "vol_20",
    "realized_vol_5",
    "realized_vol_20",
]
trade_activity_features = [
    "trade_count",
    "buy_count",
    "sell_count",
]
trade_volume_features = [
    "total_trade_volume",
    "buy_volume",
    "sell_volume",
    "avg_trade_size",
    "max_trade_size",
    "min_trade_size",
    "std_trade_size",
    "max_over_average",
]
trade_flow_features = [
    "trade_volume_imbalance",
]
trade_dynamics_features = [
    "trade_volume_change",
    "trade_count_change",
    "trade_count_imbalance",
]
lagged_trade_features = [
    "lag_trade_volume_imbalance_1",
    "lag_trade_volume_imbalance_2",
    "lag_trade_volume_imbalance_3",
    "lag_trade_volume_imbalance_5",
]

FEATURE_GROUPS = {
    "lob_features": lob_features,
    "spread": spread_features,
    "imbalance": imbalance_features,
    "liquidity": liquidity_features,
    "microprice": microprice_features,
    "returns": return_features,
    "momentum": momentum_features,
    "volatility": volatility_features,
    "trade_activity": trade_activity_features,
    "trade_volume": trade_volume_features,
    "trade_flow": trade_flow_features,
    "trade_dynamics": trade_dynamics_features,
    "lagged_trade": lagged_trade_features,
}
ALL_FEATURES = (
    lob_features
    + spread_features
    + imbalance_features
    + liquidity_features
    + microprice_features
    + return_features
    + momentum_features
    + volatility_features
    + trade_activity_features
    + trade_volume_features
    + trade_flow_features
    + trade_dynamics_features
    + lagged_trade_features
)


FEATURES_TO_TRANSFORM = [
    "lob_bids_volume_1",
    "lob_bids_volume_2",
    "lob_bids_volume_3",
    "lob_bids_volume_4",
    "lob_bids_volume_5",
    "lob_bids_volume_6",
    "lob_bids_volume_7",
    "lob_bids_volume_8",
    "lob_bids_volume_9",
    "lob_bids_volume_10",
    "lob_asks_volume_1",
    "lob_asks_volume_2",
    "lob_asks_volume_3",
    "lob_asks_volume_4",
    "lob_asks_volume_5",
    "lob_asks_volume_6",
    "lob_asks_volume_7",
    "lob_asks_volume_8",
    "lob_asks_volume_9",
    "lob_asks_volume_10",
    "rel_spread",
    "spread",
    "liquidity",
    "total_bid_volume_10",
    "total_ask_volume_10",
    "total_book_volume",
    "max_bid_ask_vol_ratio",
    "mid_minus_micro",
    "vol_5",
    "vol_20",
    "realized_vol_5",
    "realized_vol_20",
    "trade_count",
    "buy_count",
    "sell_count",
    "total_trade_volume",
    "buy_volume",
    "sell_volume",
    "avg_trade_size",
    "max_trade_size",
    "min_trade_size",
    "std_trade_size",
    "max_over_average",
]


TRANSFORMED_FEATURES = [
    "lob_bids_volume_1_log1p",
    "lob_bids_volume_2_log1p",
    "lob_bids_volume_3_log1p",
    "lob_bids_volume_4_log1p",
    "lob_bids_volume_5_log1p",
    "lob_bids_volume_6_log1p",
    "lob_bids_volume_7_log1p",
    "lob_bids_volume_8_log1p",
    "lob_bids_volume_9_log1p",
    "lob_bids_volume_10_log1p",
    "lob_asks_volume_1_log1p",
    "lob_asks_volume_2_log1p",
    "lob_asks_volume_3_log1p",
    "lob_asks_volume_4_log1p",
    "lob_asks_volume_5_log1p",
    "lob_asks_volume_6_log1p",
    "lob_asks_volume_7_log1p",
    "lob_asks_volume_8_log1p",
    "lob_asks_volume_9_log1p",
    "lob_asks_volume_10_log1p",
    "rel_spread_log1p",
    "spread_log1p",
    "liquidity_log1p",
    "total_bid_volume_10_log1p",
    "total_ask_volume_10_log1p",
    "total_book_volume_log1p",
    "max_bid_ask_vol_ratio_log1p",
    "mid_minus_micro_log1p",
    "vol_5_log1p",
    "vol_20_log1p",
    "realized_vol_5_log1p",
    "realized_vol_20_log1p",
    "trade_count_log1p",
    "buy_count_log1p",
    "sell_count_log1p",
    "total_trade_volume_log1p",
    "buy_volume_log1p",
    "sell_volume_log1p",
    "avg_trade_size_log1p",
    "max_trade_size_log1p",
    "min_trade_size_log1p",
    "std_trade_size_log1p",
    "max_over_average_log1p",
]

REMOVE_PREFIXES = (
    "lob_bids_price_",
    "lob_asks_price_",
    "lob_bids_volume_",
    "lob_asks_volume_",
)

to_transform = set(FEATURES_TO_TRANSFORM)

FINAL_FEATURES_WITH_TRANSFORM = [
    f
    for f in ALL_FEATURES
    if not f.startswith(REMOVE_PREFIXES) and f not in to_transform
] + TRANSFORMED_FEATURES


L2_FEATURE_LIST = {
    "volatility": [
        "realized_vol_20_log1p",
        "vol_20_log1p",
        "realized_vol_5_log1p",
        "vol_5_log1p",
    ],
    "trade_intensity": [
        "trade_count_log1p",
        "trade_count_change",
        "buy_count_log1p",
        "sell_count_log1p",
        "total_trade_volume_log1p",
    ],
    "trade_size": [
        "buy_volume_log1p",
        "sell_volume_log1p",
        "avg_trade_size_log1p",
        "max_trade_size_log1p",
        "min_trade_size_log1p",
        "std_trade_size_log1p",
        "max_over_average_log1p",
    ],
    "liquidity": [
        "total_ask_volume_10_log1p",
        "total_bid_volume_10_log1p",
        "total_book_volume_log1p",
        "liquidity_log1p",
        "lob_bids_volume_1_log1p",
        "lob_bids_volume_2_log1p",
        "lob_bids_volume_3_log1p",
        "lob_bids_volume_4_log1p",
        "lob_bids_volume_5_log1p",
        "lob_bids_volume_6_log1p",
        "lob_bids_volume_7_log1p",
        "lob_bids_volume_8_log1p",
        "lob_bids_volume_9_log1p",
        "lob_bids_volume_10_log1p",
    ],
    "volume_pressure": [
        "max_bid_ask_vol_ratio_log1p",
    ],
    "spread": [
        "spread_log1p",
        "rel_spread_log1p",
    ],
    "imbalance": [
        "imbalance_1",
        "imbalance_5",
        "imbalance_10",
        "imbalance_depth_1",
        "imbalance_depth_2",
        "imbalance_depth_3",
        "imbalance_depth_4",
        "imbalance_depth_5",
        "imbalance_depth_6",
        "imbalance_depth_7",
        "imbalance_depth_8",
        "imbalance_depth_9",
        "imbalance_depth_10",
    ],
    "depth_shape": [
        "lob_depth_ratio_2",
        "lob_depth_ratio_3",
        "lob_depth_ratio_4",
        "lob_depth_ratio_5",
        "lob_depth_ratio_6",
        "lob_depth_ratio_7",
        "lob_depth_ratio_8",
        "lob_depth_ratio_9",
        "lob_depth_ratio_10",
    ],
    "trade_flow": [
        "trade_volume_imbalance",
        "trade_volume_change",
        "sell_volume_log1p",
        "buy_volume_log1p",
    ],
}


MID_PRICE_MOVE_FINAL = [
    #    "mid_price_moves",
    # Vol
    "realized_vol_20_log1p",
    "realized_vol_5_log1p",
    # trade intensity
    "trade_count_log1p",
    "trade_count_change",
    # trade size
    "std_trade_size_log1p",
    "max_trade_size_log1p",
    "avg_trade_size_log1p",
    # liquidity
    "total_bid_volume_10_log1p",
    "total_book_volume_log1p",
    # volume_pressure
    "max_bid_ask_vol_ratio_log1p",
    # spread
    "rel_spread_log1p",
    # imbalance
    "imbalance_5",
    "imbalance_10",
    "imbalance_depth_3",
    # depth shape
    "lob_depth_ratio_4",
    "lob_depth_ratio_2",
    # trade_flow
    "sell_volume_log1p",
]

EDA_FEATURES = [
    "imbalance_5",
    "microprice_change",
    "microprice_weighted_10",
    "trade_volume_imbalance",
    "log_return_5",
    "realized_vol_20",
    "liquidity_log1p",
    "imbalance_depth_3",
    "imbalance_depth_5",
    "lob_depth_ratio_7",
    "trade_count_imbalance",
    "lag_trade_volume_imbalance_1",
    "std_trade_size_log1p",
    "momentum_20_log_return_1",
]


SIGN_MODEL_FEATURES = [
    #    "mid_price_change_1_sign",
    "imbalance_10",
    "imbalance_5",
    "max_bid_ask_vol_ratio_log1p",
    "total_bid_volume_10_log1p",
    "trade_count_log1p",
    "trade_count_imbalance",
    "microprice_change",
    "trade_volume_imbalance",
    "realized_vol_20_log1p",
    "trade_count_change",
    "rel_spread_log1p",
    "lob_depth_ratio_4",
    "mid_minus_micro_log1p",
]
