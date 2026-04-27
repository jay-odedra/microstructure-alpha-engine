# High-Frequency Alpha Engine

## Overview

This project builds a short-horizon alpha model using limit order book (LOB) and trade data.

The goal is to predict **next-second mid-price movements** in a high-frequency setting using market microstructure features.

The modelling approach is split into two stages:

- **Move model**: predicts whether the mid-price will change
- **Sign model**: predicts direction conditional on a move

These are combined into a final trading signal and evaluated through a simple backtest.

---

## Data

Data is sourced from the Binance exchange for the BTCUSDT trading pair.

- Order book snapshots (top 100 levels)
- Recent trades (~1000 per snapshot)
- Sampling frequency: ~1 second

### Alignment

Trades and order book data are aligned using a backward time join:

- Each trade is matched to the most recent past order book snapshot
- This ensures no forward-looking information is used

Timestamps are adjusted to account for observed latency.

---

## Feature Engineering

Features are constructed from both the order book and trade flow. (some of these are listed below)

### Price Features
- Mid-price
- Bid–ask spread
- Relative spread

### Imbalance Features
- Volume imbalance at multiple depths (levels 1, 5, 10)
- Depth ratios

### Liquidity Features
- Total bid/ask volume
- Total book volume
- Volume concentration

### Trade Flow Features
- Trade count
- Buy/sell volume
- Trade size statistics (mean, max, std)
- Volume imbalance

### Volatility & Returns
- Log returns (multiple horizons)
- Rolling volatility
- Realised volatility

Heavy-tailed features are transformed using signed log scaling where appropriate.

---

## Targets

Two prediction targets are defined:

- **Move**
  Binary variable indicating whether the mid-price changes in the next second

- **Sign**
  Direction of the price change (conditional on a move)

---

## Modelling

### Approach

Models are trained using a time-series cross-validation framework:

- `TimeSeriesSplit` with gap to prevent leakage
- Rolling training method as a very rough way to account for regime shifts (prior to explicit regime modelling)
- Out-of-fold (OOF) predictions used for evaluation

### Models

- Logistic Regression (L1, L2, Elastic Net)
- Random Forest
- Extra trees
- XGBoost

Logistic regression is used as a baseline and for feature selection.
Tree-based models capture non-linear interactions.

### Calibration

models can be calibrated to improve probability estimates (currently not due to time series nature)

---

## Strategy Construction

The final signal combines both models:

P(up) = P(move) × P(up | move)

A trading signal is constructed as:

- Long if high probability of upward move
- Short if high probability of downward move
- No trade otherwise

Thresholding is applied:

- Move probability filter (e.g. top quantiles)
- Sign confidence threshold

To avoid lookahead bias + since data granularit is ~1s (many trades couldnt happen in that time frame):

- Signals are shifted forward by one time step before applying returns

---

## Backtesting

Strategy performance is evaluated using:

- Sharpe ratio
- Total PnL
- Maximum drawdown
- Hit rate
- Information Coefficient (IC)

Transaction costs are approximated using bid–ask spread.

---

### Key Findings

**Move Model**
- Driven by:
  - volatility
  - liquidity
  - trade intensity
- Captures *when the market is active*

**Sign Model**
- Still contains signal:
  - order book imbalance
  - microprice
  - trade flow
- Captures directional pressure

**Combined Signal**
- Improves trading performance vs individual models
- Acts as:
  - move - trade filter
  - sign - direction selector
- Reduces low-quality trades

---

### Threshold Behaviour

**Move Model (quantile thresholding)**
- Optimising for **Sharpe per trade** pushes the move threshold to **low quantiles**
- This means:
  - the model assigns similar probabilities across observations
  - filtering aggressively removes too many trades without improving quality
- Interpretation:
  - move model is better at detecting **general activity regimes** than ranking *high-conviction events*

**Sign Model (probability thresholding)**
- Optimising for Sharpe per trade pushes thresholds **higher probabilities**
- This means:
  - only high-confidence directional predictions are useful
  - weak predictions add noise and reduce performance per trade
- Interpretation:
  - directional signal is **sparse but stronger when present**
  - requires **selective trading**

---

### Market Efficiency Observation

At a 1-second horizon, mid-price returns exhibit near-zero autocorrelation, consistent with martingale-like behaviour.

- Past returns contain little predictive information about future returns
- Using lagged returns as signals does not generate excess returns
- Any predictive power must therefore come from:
  - order book state
  - trade flow
  - microstructure features

---


## Regime Modelling (HMM)

Markets are non-stationary, with changing volatility, liquidity, and microstructure dynamics.
To model this explicitly, we fit a **Gaussian Hidden Markov Model (HMM)** on slow-moving structural features:

- volatility (vol_5, vol_20)
- bid–ask spread
- order book imbalance
- trade flow imbalance

The HMM infers **regime probabilities** at each timestamp, representing latent market conditions.

---

### Strategy Integration

The HMM is used as a **filtering layer on top of the alpha model**.

- Select a regime of interest (e.g. low-volatility regime)
- Apply a probability threshold:
- Only allow trades when this condition is satisfied

This gates the strategy, restricting trading to favourable market conditions.

---

### Regime-Aware Optimisation

The strategy is optimised in two stages for **Sharpe ratio**:

1. **Alpha optimisation (within regime)**
   - Fix regime \(k\)
   - Optimise:
     - move threshold (quantile)
     - sign threshold
   - Store optimal parameters

2. **Regime threshold optimisation**
   - Fix alpha thresholds
   - Vary regime probability threshold \(\theta\)
   - Select value that maximises Sharpe

---

### Key Findings

- Regimes correspond primarily to **volatility and liquidity states**
- Alpha performance is **regime-dependent**
- Filtering trades using regime probabilities:
  - reduces low-quality trades
  - improves Sharpe ratio and other metrics

---

### Interpretation

The HMM captures persistent market structure that is not directly observable from raw features.
Using it as a gating mechanism improves performance without modifying the underlying alpha model. Also a big imporvement on reproducibility across other datasets vs adhoc timeseries splits with rolling training
