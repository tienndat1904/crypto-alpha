"""
Tests for Multi-Timeframe Strategy
====================================
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from strategies.multi_timeframe import MultiTimeframeStrategy
from utils.indicators import add_all_indicators


def make_trending_up(n: int, start: float = 100) -> pd.DataFrame:
    """Create uptrending OHLCV data."""
    prices = [start + i * 0.5 for i in range(n)]
    noise = np.random.RandomState(42).normal(0, 0.2, n)
    prices = [p + n for p, n in zip(prices, noise)]
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1e6] * n,
    }, index=dates)


def make_trending_down(n: int, start: float = 200) -> pd.DataFrame:
    """Create downtrending OHLCV data."""
    prices = [start - i * 0.5 for i in range(n)]
    noise = np.random.RandomState(42).normal(0, 0.2, n)
    prices = [max(10, p + n) for p, n in zip(prices, noise)]
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1e6] * n,
    }, index=dates)


def make_4h_data(n: int = 500) -> pd.DataFrame:
    """Create 4H OHLCV with indicators."""
    rng = np.random.RandomState(42)
    returns = rng.normal(0.0002, 0.015, n)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.003, 0.003, n)),
        "high": prices * (1 + rng.uniform(0.001, 0.015, n)),
        "low": prices * (1 - rng.uniform(0.001, 0.015, n)),
        "close": prices,
        "volume": rng.uniform(1e6, 1e8, n),
    }, index=dates)
    return add_all_indicators(df).dropna()


class TestDailyTrend:

    def test_uptrend_detected(self):
        """Strong uptrend should produce bullish bias."""
        df_1d = make_trending_up(200)
        df_1d = add_all_indicators(df_1d).dropna()
        mtf = MultiTimeframeStrategy()
        trend = mtf.get_daily_trend(df_1d)

        # At least some bullish days in an uptrend
        assert (trend == 1).sum() > 0

    def test_downtrend_detected(self):
        """Strong downtrend should produce bearish bias."""
        df_1d = make_trending_down(200)
        df_1d = add_all_indicators(df_1d).dropna()
        mtf = MultiTimeframeStrategy()
        trend = mtf.get_daily_trend(df_1d)

        assert (trend == -1).sum() > 0

    def test_trend_values_valid(self):
        """Daily trend should only be -1, 0, or 1."""
        df_1d = make_trending_up(100)
        df_1d = add_all_indicators(df_1d).dropna()
        mtf = MultiTimeframeStrategy()
        trend = mtf.get_daily_trend(df_1d)

        assert set(trend.unique()).issubset({-1, 0, 1})


class TestMerge:

    def test_merge_preserves_4h_index(self):
        """Merged trend should match 4H DataFrame length."""
        df_1d = make_trending_up(200)
        df_1d = add_all_indicators(df_1d).dropna()
        df_4h = make_4h_data(500)
        mtf = MultiTimeframeStrategy()

        daily_trend = mtf.get_daily_trend(df_1d)
        trend_4h = mtf.merge_daily_trend_to_4h(df_4h, daily_trend)

        assert len(trend_4h) == len(df_4h)
        assert set(trend_4h.unique()).issubset({-1, 0, 1})


class TestSignalGeneration:

    def test_signals_valid_values(self):
        """Generated signals should be -1, 0, or 1."""
        df_4h = make_4h_data(500)
        df_1d = make_trending_up(200)
        df_1d = add_all_indicators(df_1d).dropna()

        mtf = MultiTimeframeStrategy()
        signals = mtf.generate_signals(df_4h, df_1d)

        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(signals) == len(df_4h)

    def test_no_long_in_downtrend(self):
        """Should not produce long signals when daily trend is bearish."""
        df_4h = make_4h_data(500)
        df_1d = make_trending_down(200)
        df_1d = add_all_indicators(df_1d).dropna()

        mtf = MultiTimeframeStrategy()
        daily_trend = mtf.get_daily_trend(df_1d)
        trend_4h = mtf.merge_daily_trend_to_4h(df_4h, daily_trend)
        signals = mtf.generate_signals(df_4h, df_1d)

        # Where trend is bearish, no long signals
        bearish_mask = trend_4h == -1
        if bearish_mask.any():
            long_in_bearish = (signals[bearish_mask] == 1).sum()
            assert long_in_bearish == 0, "Long signals found in bearish trend"

    def test_check_current_signal_keys(self):
        """check_current_signal should return expected keys."""
        df_4h = make_4h_data(500)
        df_1d = make_trending_up(200)
        df_1d = add_all_indicators(df_1d).dropna()

        mtf = MultiTimeframeStrategy()
        result = mtf.check_current_signal(df_4h, df_1d)

        expected = ["signal", "reason", "strategy", "daily_trend", "daily_trend_label"]
        for key in expected:
            assert key in result, f"Missing key: {key}"
        assert result["strategy"] == "multi_timeframe"
