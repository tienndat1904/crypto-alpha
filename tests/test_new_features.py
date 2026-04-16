"""
Tests for New Features
=======================
Tests for:
1. ATR-based trailing stop
2. Universe Scanner
3. MTF bias in signal generator
"""

import json
import os
import time
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

import sys
sys.path.insert(0, ".")

# Mock the database dependency before importing modules
sys.modules["pymysql"] = MagicMock()
mock_engine = MagicMock()
with patch.dict("sys.modules", {
    "data.models": MagicMock(engine=mock_engine),
}):
    from utils.correlation import CorrelationFilter

import trading.risk_manager as rm_module
from utils.universe_scanner import UniverseScanner


# ═══════════════════════════════════════════
# Helper: synthetic OHLCV data
# ═══════════════════════════════════════════

def _make_ohlcv_df(n=100, trend="up", base_price=100.0):
    """
    Generate synthetic candle data with controlled trend direction.

    Args:
        n: number of candles
        trend: "up" or "down"
        base_price: starting price
    Returns:
        DataFrame with columns: open, high, low, close, volume
        indexed by timestamp.
    """
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    prices = [base_price]
    for i in range(1, n):
        if trend == "up":
            change = np.random.normal(0.002, 0.01)
        elif trend == "down":
            change = np.random.normal(-0.002, 0.01)
        else:  # neutral / sideways
            change = np.random.normal(0.0, 0.005)
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    highs = prices * (1 + np.random.uniform(0.001, 0.02, n))
    lows = prices * (1 - np.random.uniform(0.001, 0.02, n))
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, n))
    volumes = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=timestamps)
    df.index.name = "timestamp"
    return df


def _make_ohlcv_raw(n=100, trend="up", base_price=100.0):
    """Return raw OHLCV list-of-lists as ccxt would return."""
    df = _make_ohlcv_df(n, trend, base_price)
    rows = []
    for ts, row in df.iterrows():
        rows.append([
            int(ts.timestamp() * 1000),
            row["open"], row["high"], row["low"], row["close"], row["volume"],
        ])
    return rows


# ═══════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════

@pytest.fixture
def temp_state_file(tmp_path):
    """Use a temporary state file for each test."""
    state_file = tmp_path / "state.json"
    original = rm_module.STATE_FILE
    rm_module.STATE_FILE = state_file
    yield state_file
    rm_module.STATE_FILE = original


@pytest.fixture
def risk_manager(temp_state_file):
    """Create a RiskManager with mocked correlation filter and temp state."""
    with patch.object(rm_module, "CorrelationFilter") as MockCF:
        mock_cf = MockCF.return_value
        mock_cf.can_open_position.return_value = (True, "OK")

        rm = rm_module.RiskManager()
        rm.correlation_filter = mock_cf
        yield rm


# ═══════════════════════════════════════════
# 1. TestATRTrailingStop
# ═══════════════════════════════════════════

class TestATRTrailingStop:
    """Test ATR-based trailing stop in check_stops_and_tp()."""

    def test_atr_trailing_long(self, risk_manager):
        """Long position with atr_pct=2.0: trail_pct = (2.0/100)*2 = 0.04."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03, atr_pct=2.0)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["highest_price"] = 55000  # price rallied

        # trail_pct = 0.04, trailing stop = 55000 * (1 - 0.04) = 52800
        # Price at 52700 should trigger trailing stop
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 52700})
        assert len(closed) == 1
        assert closed[0]["reason"] == "trailing_stop"

    def test_atr_trailing_long_not_triggered(self, risk_manager):
        """Long position: price above trailing stop should NOT trigger."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03, atr_pct=2.0)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["highest_price"] = 55000

        # trailing stop = 55000 * 0.96 = 52800, price at 53000 should NOT trigger
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 53000})
        assert len(closed) == 0

    def test_atr_trailing_short(self, risk_manager):
        """Short position with atr_pct=2.0: trail above lowest price."""
        risk_manager.open_position("BTC/USDT", "short", 50000, 0.03, atr_pct=2.0)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["tp2_price"] = None  # Disable TP2 so trailing stop is tested
        pos["highest_price"] = 45000  # price dropped (favorable for short)

        # trail_pct = 0.04, trailing stop = 45000 * (1 + 0.04) = 46800
        # Price at 47000 should trigger trailing stop
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 47000})
        assert len(closed) == 1
        assert closed[0]["reason"] == "trailing_stop"

    def test_atr_trailing_clamp_min(self, risk_manager):
        """Very low atr_pct=0.3: trail_pct should clamp to 0.01 (1% min)."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03, atr_pct=0.3)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["tp2_price"] = None  # Disable TP2 to isolate trailing stop
        pos["highest_price"] = 55000

        # trail_pct = (0.3/100)*2 = 0.006, clamped to 0.01
        # trailing stop = 55000 * (1 - 0.01) = 54450
        # Price at 54400 should trigger
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 54400})
        assert len(closed) == 1
        assert closed[0]["reason"] == "trailing_stop"

        # Verify price slightly above stop would NOT trigger
        risk_manager.open_position("ETH/USDT", "long", 3000, 0.03, atr_pct=0.3)
        pos2 = risk_manager.state["open_positions"]["ETH/USDT"]
        pos2["tp1_hit"] = True
        pos2["tp2_price"] = None
        pos2["highest_price"] = 3300
        # trailing stop = 3300 * 0.99 = 3267
        closed2 = risk_manager.check_stops_and_tp({"ETH/USDT": 3270})
        assert len(closed2) == 0

    def test_atr_trailing_clamp_max(self, risk_manager):
        """Very high atr_pct=5.0: trail_pct should clamp to 0.08 (8% max)."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03, atr_pct=5.0)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["tp2_price"] = None  # Disable TP2 to isolate trailing stop
        pos["highest_price"] = 60000

        # trail_pct = (5.0/100)*2 = 0.10, clamped to 0.08
        # trailing stop = 60000 * (1 - 0.08) = 55200
        # Price at 55100 should trigger
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 55100})
        assert len(closed) == 1
        assert closed[0]["reason"] == "trailing_stop"

        # Price at 55300 should NOT trigger (above 55200)
        risk_manager.open_position("ETH/USDT", "long", 3000, 0.03, atr_pct=5.0)
        pos2 = risk_manager.state["open_positions"]["ETH/USDT"]
        pos2["tp1_hit"] = True
        pos2["tp2_price"] = None
        pos2["highest_price"] = 3600
        # trailing stop = 3600 * 0.92 = 3312
        closed2 = risk_manager.check_stops_and_tp({"ETH/USDT": 3320})
        assert len(closed2) == 0

    def test_atr_trailing_fallback(self, risk_manager):
        """Position without atr_pct (None) should use stop_loss_pct as fallback."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03, atr_pct=None)
        pos = risk_manager.state["open_positions"]["BTC/USDT"]
        pos["tp1_hit"] = True
        pos["highest_price"] = 55000

        # Fallback trail_pct = stop_loss_pct = 0.03
        # trailing stop = 55000 * (1 - 0.03) = 53350
        # Price at 53300 should trigger
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 53300})
        assert len(closed) == 1
        assert closed[0]["reason"] == "trailing_stop"


# ═══════════════════════════════════════════
# 2. TestUniverseScanner
# ═══════════════════════════════════════════

class TestUniverseScanner:
    """Test UniverseScanner with mocked exchange."""

    @pytest.fixture
    def scanner(self):
        """Create a scanner with mocked exchange."""
        with patch("utils.universe_scanner.ccxt.binance") as MockExchange:
            mock_exchange = MockExchange.return_value
            scanner = UniverseScanner()
            scanner.exchange = mock_exchange
            yield scanner

    def _make_ticker(self, symbol, quote_volume=100_000_000, last=100.0,
                     bid=99.9, ask=100.1):
        """Helper to create a ticker dict."""
        return {
            "symbol": symbol,
            "quoteVolume": quote_volume,
            "last": last,
            "bid": bid,
            "ask": ask,
        }

    def test_filter_blacklisted(self, scanner):
        """USDC/USDT, BUSD/USDT etc should be filtered out."""
        tickers = {
            "USDC/USDT": self._make_ticker("USDC/USDT"),
            "BUSD/USDT": self._make_ticker("BUSD/USDT"),
            "TUSD/USDT": self._make_ticker("TUSD/USDT"),
            "FDUSD/USDT": self._make_ticker("FDUSD/USDT"),
            "BTC/USDT": self._make_ticker("BTC/USDT"),
        }
        candidates = scanner._filter_usdt_pairs(tickers)
        symbols = [c["symbol"] for c in candidates]
        assert "USDC/USDT" not in symbols
        assert "BUSD/USDT" not in symbols
        assert "TUSD/USDT" not in symbols
        assert "FDUSD/USDT" not in symbols
        assert "BTC/USDT" in symbols

    def test_filter_leveraged_tokens(self, scanner):
        """BTCUP, ETHDOWN should be filtered."""
        tickers = {
            "BTCUP/USDT": self._make_ticker("BTCUP/USDT"),
            "ETHDOWN/USDT": self._make_ticker("ETHDOWN/USDT"),
            "BNBBULL/USDT": self._make_ticker("BNBBULL/USDT"),
            "SOLBEAR/USDT": self._make_ticker("SOLBEAR/USDT"),
            "BTC/USDT": self._make_ticker("BTC/USDT"),
        }
        candidates = scanner._filter_usdt_pairs(tickers)
        symbols = [c["symbol"] for c in candidates]
        assert "BTCUP/USDT" not in symbols
        assert "ETHDOWN/USDT" not in symbols
        assert "BNBBULL/USDT" not in symbols
        assert "SOLBEAR/USDT" not in symbols
        assert "BTC/USDT" in symbols

    def test_filter_non_ascii(self, scanner):
        """Symbols with non-ASCII chars should be filtered."""
        tickers = {
            "\u5e01\u5b89/USDT": self._make_ticker("\u5e01\u5b89/USDT"),
            "BTC/USDT": self._make_ticker("BTC/USDT"),
        }
        candidates = scanner._filter_usdt_pairs(tickers)
        symbols = [c["symbol"] for c in candidates]
        assert "\u5e01\u5b89/USDT" not in symbols
        assert "BTC/USDT" in symbols

    def test_filter_stablecoins(self, scanner):
        """USD1, RLUSD, XAUT etc should be filtered."""
        stables = ["USD1", "USDE", "USDP", "DAI", "FRAX", "LUSD",
                    "PYUSD", "RLUSD", "XAUT", "PAXG"]
        tickers = {f"{s}/USDT": self._make_ticker(f"{s}/USDT") for s in stables}
        tickers["BTC/USDT"] = self._make_ticker("BTC/USDT")

        candidates = scanner._filter_usdt_pairs(tickers)
        symbols = [c["symbol"] for c in candidates]
        for s in stables:
            assert f"{s}/USDT" not in symbols
        assert "BTC/USDT" in symbols

    def test_filter_volume_threshold(self, scanner):
        """Coins below $50M volume should be filtered."""
        tickers = {
            "LOW/USDT": self._make_ticker("LOW/USDT", quote_volume=10_000_000),
            "HIGH/USDT": self._make_ticker("HIGH/USDT", quote_volume=100_000_000),
        }
        candidates = scanner._filter_usdt_pairs(tickers)
        symbols = [c["symbol"] for c in candidates]
        assert "LOW/USDT" not in symbols
        assert "HIGH/USDT" in symbols

    def test_normalize(self):
        """Test the _normalize static method with known values."""
        series = pd.Series([10, 20, 30, 40, 50])
        result = UniverseScanner._normalize(series)
        assert result.iloc[0] == 0.0
        assert result.iloc[-1] == 1.0
        assert abs(result.iloc[2] - 0.5) < 1e-10

        # All same values should return 0.5
        same = pd.Series([5, 5, 5])
        result_same = UniverseScanner._normalize(same)
        assert (result_same == 0.5).all()

    def test_generate_config_high_vol(self, scanner):
        """Mock data with high volatility: roc_threshold should be -10."""
        # Create data with high volatility (std of log returns > 0.03)
        np.random.seed(99)
        n = 42
        prices = [100.0]
        for _ in range(n - 1):
            # Large random moves => high vol
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.06)))
        prices = np.array(prices)
        timestamps = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
        highs = prices * 1.02
        lows = prices * 0.98

        raw_candles = []
        for i in range(n):
            raw_candles.append([
                int(timestamps[i].timestamp() * 1000),
                float(prices[i] * 1.001),
                float(highs[i]),
                float(lows[i]),
                float(prices[i]),
                1000.0,
            ])

        scanner.exchange.fetch_ohlcv.return_value = raw_candles
        config = scanner.generate_config("HIGHVOL/USDT")

        assert config != {}
        mr_params = config["strategies"][0]["params"]
        assert mr_params["roc_threshold"] == -10.0

    def test_generate_config_low_vol(self, scanner):
        """Mock data with low volatility: roc_threshold should be -8."""
        np.random.seed(42)
        n = 42
        prices = [100.0]
        for _ in range(n - 1):
            # Small random moves => low vol
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))
        prices = np.array(prices)
        timestamps = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
        highs = prices * 1.005
        lows = prices * 0.995

        raw_candles = []
        for i in range(n):
            raw_candles.append([
                int(timestamps[i].timestamp() * 1000),
                float(prices[i] * 1.001),
                float(highs[i]),
                float(lows[i]),
                float(prices[i]),
                1000.0,
            ])

        scanner.exchange.fetch_ohlcv.return_value = raw_candles
        config = scanner.generate_config("LOWVOL/USDT")

        assert config != {}
        mr_params = config["strategies"][0]["params"]
        assert mr_params["roc_threshold"] == -8.0


# ═══════════════════════════════════════════
# 3. TestMTFBias
# ═══════════════════════════════════════════

class TestMTFBias:
    """Test the _get_mtf_bias() method in SignalGenerator."""

    @pytest.fixture
    def signal_gen(self):
        """Create a SignalGenerator with mocked exchange."""
        with patch("trading.signal_generator.ccxt.binance") as MockExchange:
            mock_exchange = MockExchange.return_value
            from trading.signal_generator import SignalGenerator
            sg = SignalGenerator()
            sg.exchange = mock_exchange
            yield sg

    @staticmethod
    def _make_indicator_df(n=100, close=110.0, ema_21=105.0, ema_50=100.0,
                           rsi=55.0, ema_21_3ago=103.0):
        """
        Create a DataFrame with pre-set indicator values.
        Instead of relying on synthetic price data to produce the right
        EMA/RSI values, we directly set the indicator columns.
        """
        timestamps = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": [close] * n,
            "high": [close * 1.01] * n,
            "low": [close * 0.99] * n,
            "close": [close] * n,
            "volume": [1000.0] * n,
            "ema_21": [ema_21] * n,
            "ema_50": [ema_50] * n,
            "rsi": [rsi] * n,
        }, index=timestamps)
        df.index.name = "timestamp"
        # Set ema_21 3 bars ago to different value for hourly trending check
        if n >= 4:
            df.iloc[-4, df.columns.get_loc("ema_21")] = ema_21_3ago
        return df

    def _setup_mtf_fetch(self, signal_gen, daily_df, hourly_df):
        """
        Mock fetch_latest to return pre-built DataFrames, and
        mock add_all_indicators to return the df unchanged (indicators already set).
        """
        original_fetch = signal_gen.fetch_latest

        def mock_fetch_latest(symbol, timeframe, limit=250):
            if timeframe == "1d":
                return daily_df
            elif timeframe == "1h":
                return hourly_df
            else:
                return _make_ohlcv_df(n=limit)

        signal_gen.fetch_latest = MagicMock(side_effect=mock_fetch_latest)

    def test_daily_bullish(self, signal_gen):
        """When EMA21 > EMA50 and close > EMA21, daily_bias should be 1."""
        # close=110 > ema_21=105 > ema_50=100 => bullish
        daily_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100)
        hourly_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100,
                                            rsi=55, ema_21_3ago=103)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["daily_bias"] == 1
        assert result["daily_trend"] == "bullish"

    def test_daily_bearish(self, signal_gen):
        """When EMA21 < EMA50 and close < EMA21, daily_bias should be -1."""
        # close=90 < ema_21=95 < ema_50=100 => bearish
        daily_df = self._make_indicator_df(close=90, ema_21=95, ema_50=100)
        hourly_df = self._make_indicator_df(close=90, ema_21=95, ema_50=100,
                                            rsi=40, ema_21_3ago=97)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["daily_bias"] == -1
        assert result["daily_trend"] == "bearish"

    def test_daily_neutral(self, signal_gen):
        """When mixed signals, daily_bias should be 0."""
        # ema_21=105 > ema_50=100, but close=102 < ema_21=105 => mixed => neutral
        daily_df = self._make_indicator_df(close=102, ema_21=105, ema_50=100)
        hourly_df = self._make_indicator_df(close=102, ema_21=105, ema_50=100,
                                            rsi=50, ema_21_3ago=104)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["daily_bias"] == 0
        assert result["daily_trend"] == "neutral"

    def test_hourly_bullish(self, signal_gen):
        """RSI > 45, close > EMA21, EMA21 trending up => hourly_bias = 1."""
        daily_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100)
        # rsi=55 > 45, close=110 > ema_21=105, ema_21=105 > ema_21_3ago=103
        hourly_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100,
                                            rsi=55, ema_21_3ago=103)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["hourly_bias"] == 1
        assert result["hourly_trend"] == "bullish"

    def test_confirmed(self, signal_gen):
        """When both daily and hourly agree and are non-zero, confirmed=True."""
        # Both bullish
        daily_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100)
        hourly_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100,
                                            rsi=55, ema_21_3ago=103)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["daily_bias"] == 1
        assert result["hourly_bias"] == 1
        assert result["confirmed"] is True
        assert "CONFIRMED" in result["reason"]

    def test_cache_works(self, signal_gen):
        """Call twice, verify fetch_latest only called for first invocation."""
        daily_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100)
        hourly_df = self._make_indicator_df(close=110, ema_21=105, ema_50=100,
                                            rsi=55, ema_21_3ago=103)
        self._setup_mtf_fetch(signal_gen, daily_df, hourly_df)

        with patch("trading.signal_generator.add_all_indicators", side_effect=lambda df: df):
            result1 = signal_gen._get_mtf_bias("BTC/USDT")
            call_count_after_first = signal_gen.fetch_latest.call_count

            result2 = signal_gen._get_mtf_bias("BTC/USDT")
            call_count_after_second = signal_gen.fetch_latest.call_count

        # Second call should use cache, no additional fetch_latest calls
        assert call_count_after_second == call_count_after_first
        assert result1 == result2

    def test_fetch_failure_returns_neutral(self, signal_gen):
        """When fetch fails, should return neutral default."""
        signal_gen.fetch_latest = MagicMock(
            side_effect=Exception("Network error")
        )

        result = signal_gen._get_mtf_bias("BTC/USDT")
        assert result["daily_bias"] == 0
        assert result["hourly_bias"] == 0
        assert result["confirmed"] is False

    def test_signal_rejected_by_daily(self, signal_gen):
        """Generate a SHORT signal but daily is bullish => should be filtered."""
        # Pre-populate cache with bullish bias so generate_signal uses it
        signal_gen._mtf_cache["BTC/USDT"] = {
            "data": {
                "daily_bias": 1,
                "hourly_bias": 1,
                "daily_trend": "bullish",
                "hourly_trend": "bullish",
                "confirmed": True,
                "reason": "MTF: 1d=bullish, 1h=bullish, CONFIRMED",
            },
            "ts": time.time(),
        }

        # Mock fetch_latest for the main generate_signal call
        main_df = _make_ohlcv_df(n=250, trend="up")
        signal_gen.fetch_latest = MagicMock(return_value=main_df)

        # Mock the strategy check methods to return a SHORT signal
        with patch.object(signal_gen, "_check_momentum_reversal") as mock_mr, \
             patch.object(signal_gen, "_check_volatility_breakout") as mock_vb:

            mock_mr.return_value = {"signal": -1, "strategy": "momentum_reversal",
                                    "reason": "test short signal"}
            mock_vb.return_value = {"signal": -1, "strategy": "volatility_breakout",
                                    "reason": "test short signal"}

            # Mock regime detector to allow all strategies
            signal_gen.regime_detector.detect = MagicMock(return_value={
                "regime": "trending", "confidence": 0.8, "adx": 30,
            })
            signal_gen.regime_detector.get_strategy_recommendation = MagicMock(
                return_value={"avoid": []}
            )

            # Generate signal for BTC/USDT (which is in ALPHA_CONFIGS)
            result = signal_gen.generate_signal("BTC/USDT")

            # The short signal should be rejected by daily bullish bias
            # Result should be signal=0 since all strategies are filtered
            assert result["signal"] == 0
