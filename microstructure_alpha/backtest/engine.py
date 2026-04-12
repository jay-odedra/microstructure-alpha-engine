import numpy as np
import pandas as pd

from microstructure_alpha.backtest.metrics import (
    compute_combined_model_diagnostics,
    compute_strategy_performance,
    compute_trading_behaviour,
    signal_quality_metrics,
)


def apply_transaction_cost(signal, cost_per_trade):

    trades = signal.diff().abs()
    trades.iloc[0] = 0

    cost = trades * cost_per_trade
    return cost


def evaluate_strategy(
    sign_preds: pd.Series,
    move_preds: pd.Series,
    returns: pd.Series,
    move_quantile: float = 0.9,
    sign_threshold: float = 0.6,
    annualisation_factor: float = 1.0,
    transaction_cost: float = 0.0,
):
    assert sign_preds.index.equals(move_preds.index)
    assert sign_preds.index.equals(returns.index)

    signal = move_preds * (2 * sign_preds - 1)

    move_threshold = np.quantile(move_preds, move_quantile)
    move_filter = move_preds > move_threshold

    strong_up = sign_preds > sign_threshold
    strong_down = sign_preds < (1 - sign_threshold)
    sign_filter = strong_up | strong_down

    signal[~move_filter] = 0.0
    signal[~sign_filter] = 0.0

    signal = signal.shift(1)

    valid = signal.notna()
    signal = signal[valid]
    returns = returns[valid]
    assert signal.index.equals(returns.index)

    pnl = signal * returns
    costs = apply_transaction_cost(signal, transaction_cost)
    pnl = pnl - costs

    strategy_metric = compute_strategy_performance(signal, pnl)
    trading_behaviour = compute_trading_behaviour(signal, pnl)
    model_diagnostics = compute_combined_model_diagnostics(signal, returns, pnl)
    signal_quality = signal_quality_metrics(signal, returns, n_quantiles=10)

    all_metrics = {
        **strategy_metric,
        **trading_behaviour,
        **model_diagnostics,
        **signal_quality,
    }

    return all_metrics, pnl, signal
