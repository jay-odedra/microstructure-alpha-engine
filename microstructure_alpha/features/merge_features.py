import numpy as np
import pandas as pd
from microstructure_alpha.utils.constants import EPS


def merge_lob_and_trade(trade_features_df, lob_features_df):
    final_df = lob_features_df.merge(trade_features_df, on="timestamp", how="left")
    return final_df
