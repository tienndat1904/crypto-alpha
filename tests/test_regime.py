"""
Tests for Regime Detection
============================
Verifies market regime classification logic.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from utils.regime import RegimeDetector


def make_data_with_adx(adx_values: list, atr_pct: float = 0.02) -> pd.DataFrame:
    """Create minimal DataFrame with ADX and volatility indicators."""
    n = len(adx_values)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({
        "close": [100 + i * 0.1 for i in range(n)],
        "adx": adx_values,
        "atr_pct": [atr_pct] * n,
        "bb_width": [0.05] * n,
        "volatility_30d": [0.3] * n,
    }, index=dates)


class TestRegimeDetection:

    def test_high_adx_is_trending(self):
        """ADX > 40 should classify as trending with high confidence."""
        adx_values = [35.0] * 15 + [42.0] * 10
        df = make_data_with_adx(adx_values)
        rd = RegimeDetector()
        result = rd.detect(df)

        assert result["regime"] == RegimeDetector.TRENDING
        assert result["confidence"] > 0.6

    def test_low_adx_is_sideways(self):
        """ADX < 20 with normal volatility should classify as sideways."""
        adx_values = [15.0] * 25
        df = make_data_with_adx(adx_values, atr_pct=0.01)
        rd = RegimeDetector()
        result = rd.detect(df)

        assert result["regime"] in (RegimeDetector.SIDEWAYS, RegimeDetector.CHOPPY)

    def test_moderate_adx_trending(self):
        """ADX between 25-40 should classify as trending."""
        adx_values = [20.0] * 10 + [30.0] * 15
        df = make_data_with_adx(adx_values)
        rd = RegimeDetector()
        result = rd.detect(df)

        assert result["regime"] == RegimeDetector.TRENDING

    def test_result_keys(self):
        """Result should contain all expected keys."""
        adx_values = [25.0] * 25
        df = make_data_with_adx(adx_values)
        rd = RegimeDetector()
        result = rd.detect(df)

        expected_keys = [
            "regime", "confidence", "adx", "adx_mean",
            "adx_rising", "atr_pct", "bb_width", "vol_percentile"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_confidence_bounded(self):
        """Confidence should always be between 0.1 and 0.95."""
        for adx_val in [5, 15, 25, 35, 50, 70]:
            adx_values = [adx_val] * 25
            df = make_data_with_adx(adx_values)
            rd = RegimeDetector()
            result = rd.detect(df)

            assert 0.1 <= result["confidence"] <= 0.95

    def test_detect_series_returns_series(self):
        """detect_series should return a pandas Series."""
        adx_values = [15.0] * 15 + [30.0] * 15 + [10.0] * 15
        df = make_data_with_adx(adx_values)
        rd = RegimeDetector()
        regimes = rd.detect_series(df)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(df)
        assert set(regimes.unique()).issubset({
            RegimeDetector.TRENDING,
            RegimeDetector.SIDEWAYS,
            RegimeDetector.CHOPPY,
        })


class TestStrategyRecommendation:

    def test_trending_favors_trend_strategies(self):
        """Trending regime should recommend trend-following."""
        rd = RegimeDetector()
        rec = rd.get_strategy_recommendation({
            "regime": RegimeDetector.TRENDING,
            "confidence": 0.8,
        })

        assert "trend_following" in rec["preferred"]
        assert "mean_reversion" in rec["avoid"]
        assert rec["position_size_mult"] == 1.0

    def test_sideways_favors_mean_reversion(self):
        """Sideways regime should recommend mean-reversion."""
        rd = RegimeDetector()
        rec = rd.get_strategy_recommendation({
            "regime": RegimeDetector.SIDEWAYS,
            "confidence": 0.8,
        })

        assert "mean_reversion" in rec["preferred"]
        assert "trend_following" in rec["avoid"]

    def test_choppy_reduces_size(self):
        """Choppy regime should reduce position size."""
        rd = RegimeDetector()
        rec = rd.get_strategy_recommendation({
            "regime": RegimeDetector.CHOPPY,
            "confidence": 0.5,
        })

        assert rec["position_size_mult"] == 0.5

    def test_weights_sum_reasonable(self):
        """Strategy weights should be positive."""
        rd = RegimeDetector()
        for regime in [RegimeDetector.TRENDING, RegimeDetector.SIDEWAYS, RegimeDetector.CHOPPY]:
            rec = rd.get_strategy_recommendation({"regime": regime, "confidence": 0.7})
            for name, weight in rec["weights"].items():
                assert weight >= 0, f"{regime}/{name} has negative weight"
