import pandas as pd

from microstructure_alpha.backtest.engine import evaluate_strategy


def optimise_model_threshold(
    sign_preds,
    move_preds,
    returns,
    spread,
    move_quantiles,
    sign_thresholds,
    min_trade_freq=0.05,
):
    results = []

    for mq in move_quantiles:
        for st in sign_thresholds:

            all_metrics, pnl, signal_shifted, returns_shifted = evaluate_strategy(
                sign_preds=sign_preds,
                move_preds=move_preds,
                returns=returns,
                spread=spread,
                move_quantile=mq,
                sign_threshold=st,
            )

            if all_metrics["trade_freq"] < min_trade_freq:
                continue

            results.append(
                {
                    "move_q": mq,
                    "sign_th": st,
                    **all_metrics,
                }
            )

    df = pd.DataFrame(results)

    best = df.sort_values("sharpe", ascending=False).iloc[0]

    return df, best


def optimise_hmm_threshold_only(
    sign_preds,
    move_preds,
    returns,
    spread,
    hmm_preds,
    regime_idx=0,
    regime_thresholds=None,
    move_quantile=0.5,
    sign_threshold=0.5,
    min_trade_freq=0.05,
):
    results = []

    for r_th in regime_thresholds:

        mask = hmm_preds.iloc[:, regime_idx] > r_th

        move_f = move_preds.copy()
        sign_f = sign_preds.copy()

        move_f[~mask] = 0.0
        sign_f[~mask] = 0.0

        all_metrics, pnl, signal, returns_shifted = evaluate_strategy(
            sign_preds=sign_f,
            move_preds=move_f,
            returns=returns,
            spread=spread,
            move_quantile=move_quantile,
            sign_threshold=sign_threshold,
        )

        if all_metrics["trade_freq"] < min_trade_freq:
            continue

        results.append(
            {
                "regime_th": r_th,
                "regime_frac": mask.mean(),  
                **all_metrics,
            }
        )

    df = pd.DataFrame(results)
    best = df.sort_values("sharpe", ascending=False).iloc[0]

    return df, best
