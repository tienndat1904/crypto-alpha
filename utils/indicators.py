"""
Technical Indicators & Feature Engineering
============================================
Computes technical indicators and engineered features from OHLCV data.
Used in both EDA (Tuần 2) and Alpha Research (Phase 2).

Usage:
    from utils.indicators import add_all_indicators
    df = add_all_indicators(df)  # df must have: open, high, low, close, volume
"""

import numpy as np
import pandas as pd
from loguru import logger


# ═══════════════════════════════════════════
# RETURNS & VOLATILITY
# ═══════════════════════════════════════════

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log returns."""
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volatility(df: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Add rolling volatility (std of log returns) for multiple windows."""
    for w in windows:
        df[f"volatility_{w}"] = df["log_returns"].rolling(w).std()
    return df


# ═══════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════

def add_sma(df: pd.DataFrame, windows: list = [7, 20, 50, 200]) -> pd.DataFrame:
    """Simple Moving Averages."""
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list = [9, 21, 50]) -> pd.DataFrame:
    """Exponential Moving Averages."""
    for w in windows:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD, Signal line, and Histogram."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index — measures trend strength."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


# ═══════════════════════════════════════════
# MOMENTUM INDICATORS
# ═══════════════════════════════════════════

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()

    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_roc(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
    """Rate of Change — momentum measure."""
    for p in periods:
        df[f"roc_{p}"] = df["close"].pct_change(p) * 100
    return df


# ═══════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════

def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands with %B and bandwidth."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()

    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — measures volatility in price terms."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()
    df["atr_pct"] = df["atr"] / df["close"] * 100  # ATR as % of price
    return df


# ═══════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    return df


def add_volume_features(
    df: pd.DataFrame, windows: list = [7, 20]
) -> pd.DataFrame:
    """Volume SMA and relative volume."""
    for w in windows:
        df[f"volume_sma_{w}"] = df["volume"].rolling(w).mean()
    df["volume_ratio"] = df["volume"] / df[f"volume_sma_{windows[-1]}"]
    return df


def add_vwap_approx(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Approximate VWAP using rolling window."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (
        (typical_price * df["volume"]).rolling(period).sum()
        / df["volume"].rolling(period).sum()
    )
    return df


# ═══════════════════════════════════════════
# PRICE ACTION FEATURES
# ═══════════════════════════════════════════

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candlestick body size, shadow ratios, and direction."""
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body_size"] / df["candle_range"].replace(0, np.nan)
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["candle_direction"] = np.sign(df["close"] - df["open"])
    return df


def add_support_resistance(
    df: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """Rolling support (min low) and resistance (max high)."""
    df["resistance"] = df["high"].rolling(lookback).max()
    df["support"] = df["low"].rolling(lookback).min()
    df["price_position"] = (
        (df["close"] - df["support"]) / (df["resistance"] - df["support"])
    )
    return df


# ═══════════════════════════════════════════
# STATISTICAL FEATURES
# ═══════════════════════════════════════════

def add_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Z-score of price — how many std devs from rolling mean."""
    rolling_mean = df["close"].rolling(period).mean()
    rolling_std = df["close"].rolling(period).std()
    df["zscore"] = (df["close"] - rolling_mean) / rolling_std
    return df


def add_skew_kurtosis(df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
    """Rolling skewness and kurtosis of returns."""
    df["returns_skew"] = df["log_returns"].rolling(period).skew()
    df["returns_kurtosis"] = df["log_returns"].rolling(period).kurt()
    return df


# ═══════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL technical indicators and engineered features to a DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        DataFrame with ~50+ additional feature columns
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Returns & volatility
    df = add_returns(df)
    df = add_volatility(df)

    # Trend
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_adx(df)

    # Momentum
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_roc(df)

    # Volatility
    df = add_bollinger_bands(df)
    df = add_atr(df)

    # Volume
    df = add_obv(df)
    df = add_volume_features(df)
    df = add_vwap_approx(df)

    # Price action
    df = add_candle_features(df)
    df = add_support_resistance(df)

    # Statistical
    df = add_zscore(df)
    df = add_skew_kurtosis(df)

    feature_count = len(df.columns) - len(required) - 1  # -1 for timestamp
    logger.info(f"Added {feature_count} indicator/feature columns.")

    return df
