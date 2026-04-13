# TODO — High-Frequency Alpha Engine

## Data

- [ ] move to websocket API
- [ ] capture event-level data instead of 1s snapshots
- [ ] improve timestamp accuracy and alignment (comes with websocket)

## Features

- [ ] add spread dynamics (spread change, rolling spread)
- [ ] add exponential decay features

## Modelling

- [ ] add time-decay weighting
- [ ] run hyperparameter optimisation using time series cross-validation
- [ ] compare model classes (logreg, RF, XGB, LightGBM) on both AUC and trading metrics
- [ ] evaluate calibration quality (probability vs realised frequency)
- [ ] reduce feature set based on stability and redundancy
- [ ] more general thought put into models


## Regime Modelling

- [ ] implement HMM for regime detection (possibly other models for simplicity)
- [ ] replace rolling retraining with regime conditioning
- [ ] analyse regime transitions and durations
- [ ] condition signals on regime

## Strategy

- [ ] improve signal construction (less hard thresholding)
- [ ] compare quantile filtering vs top-k selection
- [ ] add simple position sizing
- [ ] test slightly longer horizons

## Transaction Costs & Execution

- [ ] model slippage more realistically
- [ ] add latency assumptions
- [ ] analyse sensitivity to costs
- [ ] reduce overtrading

## Evaluation

- [ ] add autocorrelation-adjusted Sharpe
- [ ] check turnover
- [ ] plot equity curve and drawdowns
- [ ] test performance at different thresholds
- [ ] compare trade quality vs frequency
- [ ] check stability over time

## Robustness

- [ ] test across different time periods
- [ ] check performance across regimes
- [ ] validate signal stability

## Cleanup

- [ ] create final end-to-end notebook
- [ ] make repo clean and easy to follow
- [ ] create class structure for backtesting engine
- [ ] module needs to be much more user friendly
- [ ] comment code and documentation
- [ ] mode to config based setup yml

