"""
Tests for BacktestEngine
=========================
Verifies equity calculation, fee/slippage handling, stop-loss, and metrics.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from backtest import BacktestEngine, walk_forward_split, rolling_walk_forward


def make_ohlcv(prices: list, volume: float = 1000.0) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h")
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [volume] * n,
    }, index=dates)
    return df


class TestBacktestEngine:
    """Core backtest engine tests."""

    def test_buy_and_hold_matches_market(self):
        """A signal that's always 1 should track the market (minus costs)."""
        prices = [100, 102, 104, 103, 106]
        df = make_ohlcv(prices)
        signals = pd.Series(1, index=df.index)

        bt = BacktestEngine(df, initial_capital=1000, fee=0, slippage_pct=0)
        bt.run(signals)
        metrics = bt.get_metrics()

        # With no costs, strategy return ≈ market return
        assert abs(metrics["total_return_pct"] - metrics["buy_hold_return_pct"]) < 1.0

    def test_flat_signal_no_return(self):
        """A signal that's always 0 should produce 0 return."""
        prices = [100, 110, 90, 105, 100]
        df = make_ohlcv(prices)
        signals = pd.Series(0, index=df.index)

        bt = BacktestEngine(df, initial_capital=1000, fee=0.001, slippage_pct=0.0005)
        bt.run(signals)
        metrics = bt.get_metrics()

        assert metrics["total_return_pct"] == 0.0
        assert metrics["total_trades"] == 0

    def test_fees_reduce_returns(self):
        """Higher fees should produce lower returns."""
        prices = [100, 102, 104, 102, 106, 108]
        df = make_ohlcv(prices)
        signals = pd.Series([0, 1, 1, 0, 1, 1], index=df.index)

        bt_low = BacktestEngine(df, initial_capital=1000, fee=0.001, slippage_pct=0)
        bt_low.run(signals)

        bt_high = BacktestEngine(df, initial_capital=1000, fee=0.01, slippage_pct=0)
        bt_high.run(signals)

        assert bt_low.get_metrics()["final_equity"] > bt_high.get_metrics()["final_equity"]

    def test_slippage_reduces_returns(self):
        """Adding slippage should reduce returns compared to no slippage."""
        prices = [100, 102, 104, 102, 106, 108]
        df = make_ohlcv(prices)
        signals = pd.Series([0, 1, 1, 0, 1, 1], index=df.index)

        bt_no_slip = BacktestEngine(df, initial_capital=1000, fee=0.001, slippage_pct=0)
        bt_no_slip.run(signals)

        bt_slip = BacktestEngine(df, initial_capital=1000, fee=0.001, slippage_pct=0.005)
        bt_slip.run(signals)

        assert bt_no_slip.get_metrics()["final_equity"] > bt_slip.get_metrics()["final_equity"]

    def test_slippage_tracked_separately(self):
        """Metrics should report fee and slippage costs separately."""
        prices = [100, 105, 110, 105, 100]
        df = make_ohlcv(prices)
        signals = pd.Series([0, 1, 1, 0, 0], index=df.index)

        bt = BacktestEngine(df, initial_capital=1000, fee=0.001, slippage_pct=0.002)
        bt.run(signals)
        m = bt.get_metrics()

        assert m["total_fees_usd"] > 0
        assert m["total_slippage_usd"] > 0
        assert abs(m["total_costs_usd"] - m["total_fees_usd"] - m["total_slippage_usd"]) < 0.01

    def test_stop_loss_limits_drawdown(self):
        """Stop-loss should cap losses vs no stop-loss."""
        # Price drops steadily — stop-loss should limit damage
        prices = [100, 100, 95, 90, 85, 80, 75, 70]
        df = make_ohlcv(prices)
        df["atr"] = 5.0
        signals = pd.Series(1, index=df.index)

        bt_no_sl = BacktestEngine(df, initial_capital=1000, fee=0, slippage_pct=0)
        bt_no_sl.run(signals)

        bt_sl = BacktestEngine(df, initial_capital=1000, fee=0, slippage_pct=0, stop_loss_pct=0.05)
        bt_sl.run(signals)

        # With stop-loss, drawdown should be less severe
        assert bt_sl.get_metrics()["max_drawdown_pct"] > bt_no_sl.get_metrics()["max_drawdown_pct"]

    def test_short_signal_profits_on_decline(self):
        """A short signal should profit when prices decline."""
        prices = [100, 100, 95, 90, 85, 80]
        df = make_ohlcv(prices)
        signals = pd.Series([0, -1, -1, -1, -1, -1], index=df.index)

        bt = BacktestEngine(df, initial_capital=1000, fee=0, slippage_pct=0)
        bt.run(signals)

        assert bt.get_metrics()["total_return_pct"] > 0

    def test_no_look_ahead_bias(self):
        """Position at time t should use return at t+1 (shift by 1)."""
        prices = [100, 110, 100, 110, 100]  # Alternating
        df = make_ohlcv(prices)
        # Signal: buy at index 0, but return should come from index 1
        signals = pd.Series([1, 0, 0, 0, 0], index=df.index)

        bt = BacktestEngine(df, initial_capital=1000, fee=0, slippage_pct=0)
        results = bt.run(signals)

        # Position shifted by 1: signal at t=0 means position active for return at t=1
        # Return at t=1 = (110-100)/100 = 10%
        assert bt.get_metrics()["total_return_pct"] > 0

    def test_metrics_keys_complete(self):
        """get_metrics() should return all expected keys."""
        prices = [100, 102, 101, 103, 105]
        df = make_ohlcv(prices)
        signals = pd.Series([0, 1, 1, 0, 0], index=df.index)

        bt = BacktestEngine(df, initial_capital=500, fee=0.001, slippage_pct=0.0005)
        bt.run(signals)
        m = bt.get_metrics()

        expected_keys = [
            "total_return_pct", "buy_hold_return_pct", "alpha_pct",
            "ann_return_pct", "ann_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown_pct", "total_trades",
            "win_rate_pct", "avg_win_pct", "avg_loss_pct",
            "profit_factor", "expectancy_pct",
            "total_costs_usd", "total_fees_usd", "total_slippage_usd",
            "final_equity",
        ]
        for key in expected_keys:
            assert key in m, f"Missing key: {key}"


class TestWalkForward:
    """Tests for walk-forward splitting."""

    def test_single_split_covers_all_data(self):
        """Train + test should cover all data."""
        prices = list(range(100, 200))
        df = make_ohlcv(prices)
        train, test = walk_forward_split(df, train_ratio=0.7)

        assert len(train) + len(test) == len(df)
        assert len(train) == 70
        assert len(test) == 30

    def test_no_overlap(self):
        """Train and test sets should not overlap."""
        prices = list(range(100, 200))
        df = make_ohlcv(prices)
        train, test = walk_forward_split(df, train_ratio=0.7)

        assert train.index[-1] < test.index[0]

    def test_rolling_produces_multiple_folds(self):
        """rolling_walk_forward should produce multiple folds with enough data."""
        prices = list(range(100, 1600))  # 1500 rows — enough for multiple folds
        df = make_ohlcv(prices)
        folds = rolling_walk_forward(df, n_splits=4, train_ratio=0.7)

        assert len(folds) >= 2
        for train, test, info in folds:
            assert len(train) > 0
            assert len(test) > 0
            assert train.index[-1] < test.index[0]

    def test_rolling_folds_no_test_overlap(self):
        """Each fold's test set should start after its train set."""
        prices = list(range(100, 600))
        df = make_ohlcv(prices)
        folds = rolling_walk_forward(df, n_splits=5, train_ratio=0.7)

        for train, test, info in folds:
            assert train.index[-1] < test.index[0], (
                f"Fold {info['fold']}: train ends at {train.index[-1]}, "
                f"test starts at {test.index[0]}"
            )

    def test_rolling_fold_info_metadata(self):
        """Fold info should contain correct metadata."""
        prices = list(range(100, 600))
        df = make_ohlcv(prices)
        folds = rolling_walk_forward(df, n_splits=3, train_ratio=0.7)

        for train, test, info in folds:
            assert "fold" in info
            assert "train_rows" in info
            assert "test_rows" in info
            assert info["train_rows"] == len(train)
            assert info["test_rows"] == len(test)

    def test_small_dataset_fallback(self):
        """With very small data, should fall back gracefully."""
        prices = list(range(100, 120))  # Only 20 rows
        df = make_ohlcv(prices)
        folds = rolling_walk_forward(df, n_splits=5, train_ratio=0.7, min_train_rows=10)

        assert len(folds) >= 1
        for train, test, _ in folds:
            assert len(train) >= 10
