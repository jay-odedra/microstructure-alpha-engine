import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_strategy_performance(signal: pd.Series, pnl: pd.Series):
    mean_return = pnl.mean()
    volatility = pnl.std()
    total_pnl = pnl.sum()

    sharpe = mean_return / volatility if volatility != 0 else np.nan
    cum_pnl = pnl.cumsum()

    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    trade_mask = signal != 0
    hit_rate_all = (pnl > 0).mean()
    hit_rate_trades = (pnl[trade_mask] > 0).mean()
    pnl_trades = pnl[signal != 0]
    sharpe_per_trade = (
        pnl_trades.mean() / pnl_trades.std() if pnl_trades.std() != 0 else np.nan
    )

    return {
        "sharpe": sharpe,
        "sharpe_per_trade": sharpe_per_trade,
        "total_pnl": total_pnl,
        "mean_return": mean_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "hit_rate_all": hit_rate_all,
        "hit_rate_trades": hit_rate_trades,
    }


def compute_trading_behaviour(signal: pd.Series, pnl: pd.Series):
    trade_mask = signal != 0

    trade_freq = trade_mask.mean()

    pnl_trade = pnl[trade_mask]

    avg_trade_pnl = pnl_trade.mean()

    trade_vol = pnl_trade.std()
    trade_sharpe = avg_trade_pnl / trade_vol if trade_vol != 0 else np.nan

    abs_signal = signal.abs()
    avg_position = abs_signal.mean()
    max_position = abs_signal.max()

    return {
        "trade_freq": trade_freq,
        "avg_trade_pnl": avg_trade_pnl,
        "trade_sharpe": trade_sharpe,
        "avg_position": avg_position,
        "max_position": max_position,
    }


def compute_combined_model_diagnostics(
    signal: pd.Series,
    returns: pd.Series,
    pnl: pd.Series,
):

    trade_mask = signal != 0
    non_zero_mask = returns != 0

    pnl_nz = pnl[non_zero_mask]
    avg_pnl_nz = pnl_nz.mean()

    pred_direction = np.sign(signal)
    true_direction = np.sign(returns)

    valid_dir = non_zero_mask & (signal != 0)

    accuracy_nz = (pred_direction[valid_dir] == true_direction[valid_dir]).mean()

    move_coverage = (trade_mask & non_zero_mask).sum() / max(non_zero_mask.sum(), 1)

    wasted_trades = (trade_mask & ~non_zero_mask).mean()

    missed_moves = (~trade_mask & non_zero_mask).mean()

    return {
        "avg_pnl_when_move": avg_pnl_nz,
        "directional_accuracy_when_move": accuracy_nz,
        "move_coverage": move_coverage,
        "wasted_trades": wasted_trades,
        "missed_moves": missed_moves,
    }


def signal_quality_metrics(signal, returns, n_quantiles=10):

    df = pd.concat([signal, returns], axis=1).dropna()
    s = df.iloc[:, 0]
    r = df.iloc[:, 1]

    ic = spearmanr(s, r).correlation

    nz_mask = r != 0

    ic_nz = spearmanr(s[nz_mask], r[nz_mask]).correlation

    return {
        "ic": ic,
        "ic_non_zero": ic_nz,
    }
