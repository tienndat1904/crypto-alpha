"""
Technical Alpha Strategies
===========================
5 alpha strategies based on EDA insights from Phase 1.
Each returns a signal Series: 1 = long, -1 = short, 0 = flat.

Strategies:
1. Mean-Reversion (RSI + Bollinger Bands)
2. Volatility Breakout (ATR + Volume spike)
3. Trend-Following (EMA crossover + ADX filter)
4. Momentum Reversal (ROC extreme + support/resistance)
5. Composite Signal (weighted combination)
"""

import numpy as np
import pandas as pd
from utils.indicators import add_all_indicators


def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators if not present."""
    if "rsi" not in df.columns:
        df = add_all_indicators(df)
    return df


# ═══════════════════════════════════════════
# ALPHA 1: Mean-Reversion (RSI + Bollinger)
# ═══════════════════════════════════════════

def alpha_mean_reversion(
    df: pd.DataFrame,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    bb_lower_threshold: float = 0.05,
    bb_upper_threshold: float = 0.95,
) -> pd.Series:
    """
    Mean-reversion strategy:
    - BUY when RSI < oversold AND price near lower Bollinger Band
    - SELL when RSI > overbought AND price near upper Bollinger Band
    - FLAT otherwise

    EDA insight: ROC has negative correlation with forward returns,
    suggesting prices tend to revert on 4H timeframe.
    """
    df = ensure_indicators(df)

    buy = (df["rsi"] < rsi_oversold) & (df["bb_pct_b"] < bb_lower_threshold)
    sell = (df["rsi"] > rsi_overbought) & (df["bb_pct_b"] > bb_upper_threshold)

    signal = pd.Series(0, index=df.index, dtype=int)

    # State machine: hold position until opposite signal
    position = 0
    for i in range(len(df)):
        if buy.iloc[i] and position <= 0:
            position = 1
        elif sell.iloc[i] and position >= 0:
            position = -1
        signal.iloc[i] = position

    return signal


# ═══════════════════════════════════════════
# ALPHA 2: Volatility Breakout (ATR + Volume)
# ═══════════════════════════════════════════

def alpha_volatility_breakout(
    df: pd.DataFrame,
    atr_multiplier: float = 1.5,
    volume_threshold: float = 1.5,
    holding_periods: int = 6,
) -> pd.Series:
    """
    Volatility breakout strategy:
    - BUY when price breaks above (SMA + ATR * multiplier)
      AND volume > threshold * average volume
    - Hold for N periods, then exit
    - SHORT on downside breakout (symmetric)

    EDA insight: atr_pct has positive correlation with forward returns,
    suggesting high-volatility moves tend to continue.
    """
    df = ensure_indicators(df)

    upper_band = df["sma_20"] + atr_multiplier * df["atr"]
    lower_band = df["sma_20"] - atr_multiplier * df["atr"]
    vol_spike = df["volume_ratio"] > volume_threshold

    breakout_up = (df["close"] > upper_band) & vol_spike
    breakout_down = (df["close"] < lower_band) & vol_spike

    signal = pd.Series(0, index=df.index, dtype=int)
    hold_counter = 0
    position = 0

    for i in range(len(df)):
        if hold_counter > 0:
            hold_counter -= 1
            signal.iloc[i] = position
            if hold_counter == 0:
                position = 0
            continue

        if breakout_up.iloc[i]:
            position = 1
            hold_counter = holding_periods
        elif breakout_down.iloc[i]:
            position = -1
            hold_counter = holding_periods

        signal.iloc[i] = position

    return signal


# ═══════════════════════════════════════════
# ALPHA 3: Trend-Following (EMA Cross + ADX)
# ═══════════════════════════════════════════

def alpha_trend_following(
    df: pd.DataFrame,
    fast_ema: str = "ema_9",
    slow_ema: str = "ema_21",
    adx_threshold: float = 25,
) -> pd.Series:
    """
    Trend-following strategy:
    - BUY when fast EMA crosses above slow EMA AND ADX > threshold
    - SELL when fast EMA crosses below slow EMA AND ADX > threshold
    - FLAT when ADX < threshold (no trend, don't trade)

    Classic strategy with ADX filter to avoid whipsaws in
    choppy/sideways markets.
    """
    df = ensure_indicators(df)

    ema_fast = df[fast_ema]
    ema_slow = df[slow_ema]

    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    trend_strong = df["adx"] > adx_threshold

    signal = pd.Series(0, index=df.index, dtype=int)
    position = 0

    for i in range(len(df)):
        if cross_up.iloc[i] and trend_strong.iloc[i]:
            position = 1
        elif cross_down.iloc[i] and trend_strong.iloc[i]:
            position = -1
        elif not trend_strong.iloc[i] and position != 0:
            # ADX dropped below threshold, exit
            position = 0
        signal.iloc[i] = position

    return signal


# ═══════════════════════════════════════════
# ALPHA 4: Momentum Reversal (ROC + S/R)
# ═══════════════════════════════════════════

def alpha_momentum_reversal(
    df: pd.DataFrame,
    roc_threshold: float = -8.0,
    roc_exit: float = 2.0,
    support_buffer: float = 0.02,
) -> pd.Series:
    """
    Contrarian momentum strategy:
    - BUY when ROC drops sharply (< threshold) AND price is near support
      → bet on bounce from oversold
    - EXIT when ROC recovers above exit threshold
    - SHORT when ROC spikes (> |threshold|) AND price near resistance

    EDA insight: Extreme ROC values tend to revert,
    especially when price is near support/resistance levels.
    """
    df = ensure_indicators(df)

    near_support = df["price_position"] < support_buffer + 0.1
    near_resistance = df["price_position"] > 1 - support_buffer - 0.1

    oversold = (df["roc_10"] < roc_threshold) & near_support
    overbought = (df["roc_10"] > abs(roc_threshold)) & near_resistance

    signal = pd.Series(0, index=df.index, dtype=int)
    position = 0

    for i in range(len(df)):
        if oversold.iloc[i] and position <= 0:
            position = 1
        elif overbought.iloc[i] and position >= 0:
            position = -1
        elif position == 1 and df["roc_10"].iloc[i] > roc_exit:
            position = 0
        elif position == -1 and df["roc_10"].iloc[i] < -roc_exit:
            position = 0
        signal.iloc[i] = position

    return signal


# ═══════════════════════════════════════════
# ALPHA 5: Composite Signal
# ═══════════════════════════════════════════

def alpha_composite(
    df: pd.DataFrame,
    weights: dict = None,
    threshold: float = 0.3,
) -> pd.Series:
    """
    Composite alpha — weighted combination of all 4 strategies.
    
    Combines weak signals into a stronger signal via voting.
    Only enters when enough signals agree (above threshold).

    Args:
        weights: Dict of strategy weights (default equal)
        threshold: Minimum weighted score to enter (0-1)
    """
    if weights is None:
        weights = {
            "mean_reversion": 0.30,
            "volatility_breakout": 0.20,
            "trend_following": 0.25,
            "momentum_reversal": 0.25,
        }

    # Get individual signals
    s1 = alpha_mean_reversion(df)
    s2 = alpha_volatility_breakout(df)
    s3 = alpha_trend_following(df)
    s4 = alpha_momentum_reversal(df)

    # Weighted score
    score = (
        weights["mean_reversion"] * s1
        + weights["volatility_breakout"] * s2
        + weights["trend_following"] * s3
        + weights["momentum_reversal"] * s4
    )

    # Generate signal based on threshold
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[score > threshold] = 1
    signal[score < -threshold] = -1

    return signal


# ═══════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════

STRATEGIES = {
    "mean_reversion": {
        "func": alpha_mean_reversion,
        "name": "Mean-Reversion (RSI + BB)",
        "description": "Buy oversold + lower BB, sell overbought + upper BB",
    },
    "volatility_breakout": {
        "func": alpha_volatility_breakout,
        "name": "Volatility Breakout (ATR + Vol)",
        "description": "Buy/sell on ATR breakout with volume confirmation",
    },
    "trend_following": {
        "func": alpha_trend_following,
        "name": "Trend-Following (EMA + ADX)",
        "description": "EMA crossover with ADX trend filter",
    },
    "momentum_reversal": {
        "func": alpha_momentum_reversal,
        "name": "Momentum Reversal (ROC + S/R)",
        "description": "Contrarian play on extreme ROC near support/resistance",
    },
    "composite": {
        "func": alpha_composite,
        "name": "Composite Signal",
        "description": "Weighted combination of all strategies",
    },
}

# Multi-timeframe is registered separately since it needs 2 DataFrames (4H + 1D).
# Import it via: from strategies.multi_timeframe import MultiTimeframeStrategy
