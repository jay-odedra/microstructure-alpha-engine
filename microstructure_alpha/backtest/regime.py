import pandas as pd
from prettytable import PrettyTable

from microstructure_alpha.backtest.metrics import (
    compute_combined_model_diagnostics,
    compute_strategy_performance,
    compute_trading_behaviour,
    signal_quality_metrics,
)


def evaluate_by_regime(
    pnl: pd.Series,
    signal: pd.Series,
    returns: pd.Series,
    regime_probs: pd.DataFrame,
    threshold: float = 0.6,
):
    results = {}

    valid = signal.notna()

    signal_valid = signal[valid]
    returns_valid = returns[valid]

    for k in range(regime_probs.shape[1]):
        p = regime_probs.iloc[:, k]
        mask = p > threshold
        signal_r = signal_valid.copy()
        signal_r[~mask] = 0.0

        returns_r = returns_valid
        pnl_r = signal_r * returns_r

        strategy_metric = compute_strategy_performance(signal_r, pnl_r)
        trading_behaviour = compute_trading_behaviour(signal_r, pnl_r)
        model_diagnostics = compute_combined_model_diagnostics(
            signal_r, returns_r, pnl_r
        )
        signal_quality = signal_quality_metrics(signal_r, returns_r, n_quantiles=10)

        all_metrics = {
            **strategy_metric,
            **trading_behaviour,
            **model_diagnostics,
            **signal_quality,
        }
        results[f"{k}"] = all_metrics

    return results


def regime_stats_to_table(regime_stats, round_digits=4, exclude=("quantile_means",)):
    metrics = list(next(iter(regime_stats.values())).keys())

    table = PrettyTable()
    table.field_names = ["Metric"] + list(regime_stats.keys())

    for metric in metrics:
        if metric in exclude:
            continue

        row = [metric]

        for regime in regime_stats:
            val = regime_stats[regime].get(metric, None)

            if isinstance(val, (int, float)):
                row.append(round(val, round_digits))
            else:
                row.append(val)

        table.add_row(row)

    return table
