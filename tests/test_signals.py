"""
Tests for Signal Generation
=============================
Verifies each strategy produces correct entry/exit signals.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from utils.indicators import add_all_indicators
from strategies.technical_alphas import (
    alpha_mean_reversion,
    alpha_volatility_breakout,
    alpha_trend_following,
    alpha_momentum_reversal,
    alpha_composite,
    STRATEGIES,
)


def make_market_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-looking OHLCV data for testing."""
    rng = np.random.RandomState(seed)

    # Random walk with drift
    returns = rng.normal(0.0002, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)

    dates = pd.date_range("2023-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": prices * (1 + rng.uniform(0.001, 0.02, n)),
        "low": prices * (1 - rng.uniform(0.001, 0.02, n)),
        "close": prices,
        "volume": rng.uniform(1e6, 1e8, n),
    }, index=dates)

    df = add_all_indicators(df).dropna()
    return df


class TestSignalProperties:
    """All strategies should satisfy these basic properties."""

    @pytest.fixture
    def market_data(self):
        return make_market_data()

    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_signal_values_valid(self, market_data, strategy_key):
        """Signals should only be -1, 0, or 1."""
        func = STRATEGIES[strategy_key]["func"]
        signals = func(market_data)

        unique = set(signals.unique())
        assert unique.issubset({-1, 0, 1}), f"Invalid signal values: {unique}"

    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_signal_length_matches_data(self, market_data, strategy_key):
        """Signal series should have same length as input data."""
        func = STRATEGIES[strategy_key]["func"]
        signals = func(market_data)

        assert len(signals) == len(market_data)

    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_signal_index_aligned(self, market_data, strategy_key):
        """Signal index should match data index."""
        func = STRATEGIES[strategy_key]["func"]
        signals = func(market_data)

        assert signals.index.equals(market_data.index)

    @pytest.mark.parametrize("strategy_key", list(STRATEGIES.keys()))
    def test_no_nan_signals(self, market_data, strategy_key):
        """Signals should not contain NaN."""
        func = STRATEGIES[strategy_key]["func"]
        signals = func(market_data)

        assert not signals.isna().any(), "Signal contains NaN"

    @pytest.mark.parametrize("strategy_key", [
        "mean_reversion", "volatility_breakout", "momentum_reversal", "composite"
    ])
    def test_not_all_flat(self, market_data, strategy_key):
        """Strategy should produce at least some non-zero signals on random data.
        Note: trend_following excluded — random data often has ADX < 25 throughout."""
        func = STRATEGIES[strategy_key]["func"]
        signals = func(market_data)

        non_zero_pct = (signals != 0).mean()
        assert non_zero_pct > 0.0, "Strategy produced no signals at all"


class TestMeanReversion:
    """Tests specific to mean-reversion strategy."""

    def test_buys_on_oversold(self):
        """Should produce buy signal when RSI < 30 and price near lower BB."""
        df = make_market_data(n=1000, seed=123)
        signals = alpha_mean_reversion(df)

        # Find where RSI < 30 and bb_pct_b < 0.05
        oversold = (df["rsi"] < 30) & (df["bb_pct_b"] < 0.05)
        if oversold.any():
            # After an oversold condition, signal should eventually become 1
            first_oversold_idx = oversold.idxmax()
            pos = oversold.index.get_loc(first_oversold_idx)
            # Check that signal at or after oversold point is 1
            assert signals.iloc[pos] == 1


class TestMomentumReversal:
    """Tests specific to momentum reversal strategy."""

    def test_custom_parameters(self):
        """Should respect custom ROC threshold parameters."""
        df = make_market_data()

        # Strict threshold should produce fewer signals
        signals_strict = alpha_momentum_reversal(df, roc_threshold=-15)
        signals_loose = alpha_momentum_reversal(df, roc_threshold=-3)

        strict_trades = (signals_strict.diff().abs() > 0).sum()
        loose_trades = (signals_loose.diff().abs() > 0).sum()

        assert loose_trades >= strict_trades


class TestTrendFollowing:
    """Tests specific to trend-following strategy."""

    def test_no_trades_in_low_adx(self):
        """Should not trade when ADX is below threshold."""
        df = make_market_data()

        # Very high ADX threshold should produce no signals
        signals = alpha_trend_following(df, adx_threshold=100)
        assert (signals == 0).all()


class TestComposite:
    """Tests specific to composite strategy."""

    def test_high_threshold_fewer_signals(self):
        """Higher threshold should produce fewer signals."""
        df = make_market_data()

        signals_low = alpha_composite(df, threshold=0.1)
        signals_high = alpha_composite(df, threshold=0.9)

        low_active = (signals_low != 0).sum()
        high_active = (signals_high != 0).sum()

        assert low_active >= high_active


class TestStrategyRegistry:
    """Test the STRATEGIES dict is complete and consistent."""

    def test_all_strategies_have_required_keys(self):
        for key, info in STRATEGIES.items():
            assert "func" in info, f"{key} missing 'func'"
            assert "name" in info, f"{key} missing 'name'"
            assert callable(info["func"]), f"{key} func not callable"

    def test_strategy_count(self):
        assert len(STRATEGIES) == 8
