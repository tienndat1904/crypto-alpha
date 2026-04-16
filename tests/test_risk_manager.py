"""
Tests for Risk Manager
========================
Verifies position sizing, kill-switch, consecutive loss handling,
and correlation filter integration.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, ".")

# Mock the database dependency before importing risk_manager
sys.modules["pymysql"] = MagicMock()
mock_engine = MagicMock()
with patch.dict("sys.modules", {
    "data.models": MagicMock(engine=mock_engine),
}):
    from utils.correlation import CorrelationFilter

import trading.risk_manager as rm_module


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


class TestPositionSizing:
    """Test position size calculations."""

    def test_basic_position_size(self, risk_manager):
        """Position size should respect max risk per trade."""
        sizing = risk_manager.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_pct=0.03,
        )

        assert sizing["size_usdt"] > 0
        assert sizing["entry_price"] == 50000
        assert sizing["stop_loss_pct"] == 0.03
        assert sizing["stop_price"] == 50000 * 0.97

    def test_position_capped_at_40pct(self, risk_manager):
        """Position should not exceed 40% of capital."""
        # With very tight stop-loss, Kelly sizing would suggest very large position
        sizing = risk_manager.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=50000,
            stop_loss_pct=0.001,  # Very tight stop
        )

        capital = risk_manager.state["capital"]
        assert sizing["size_usdt"] <= capital * 0.4 + 0.01

    def test_reduced_size_after_3_losses(self, risk_manager):
        """Position size should halve after 3 consecutive losses."""
        sizing_normal = risk_manager.calculate_position_size(
            "BTC/USDT", 50000, 0.03
        )

        risk_manager.state["consecutive_losses"] = 3

        sizing_reduced = risk_manager.calculate_position_size(
            "BTC/USDT", 50000, 0.03
        )

        # Should be roughly 50% of normal
        assert sizing_reduced["risk_amount"] < sizing_normal["risk_amount"]
        assert abs(sizing_reduced["risk_amount"] - sizing_normal["risk_amount"] * 0.5) < 0.1


class TestCanTrade:
    """Test trading permission checks."""

    def test_can_trade_default(self, risk_manager):
        """Should be able to trade by default."""
        allowed, reason = risk_manager.can_trade()
        assert allowed is True

    def test_blocked_at_max_positions(self, risk_manager):
        """Should block when max positions reached."""
        # Fill up positions
        risk_manager.state["open_positions"] = {
            "BTC/USDT": {"size_usdt": 100},
            "ETH/USDT": {"size_usdt": 100},
            "SOL/USDT": {"size_usdt": 100},
        }

        allowed, reason = risk_manager.can_trade()
        assert allowed is False
        assert "Max positions" in reason

    def test_kill_switch_on_drawdown(self, risk_manager):
        """Should trigger kill switch on excessive drawdown."""
        risk_manager.state["peak_capital"] = 500
        risk_manager.state["capital"] = 400  # -20% drawdown

        allowed, reason = risk_manager.can_trade()
        assert allowed is False
        assert "KILL SWITCH" in reason

    def test_pause_after_5_losses(self, risk_manager):
        """Should pause trading after 5 consecutive losses."""
        risk_manager.state["consecutive_losses"] = 5

        allowed, reason = risk_manager.can_trade()
        assert allowed is False
        assert "consecutive losses" in reason.lower() or "Pausing" in reason

    def test_correlation_filter_blocks(self, risk_manager):
        """Should block when correlation filter says no."""
        risk_manager.state["open_positions"] = {
            "BTC/USDT": {"size_usdt": 100},
        }
        risk_manager.correlation_filter.can_open_position.return_value = (
            False, "Blocked: ETH/USDT too correlated with BTC/USDT (r=0.92)"
        )

        allowed, reason = risk_manager.can_trade(symbol="ETH/USDT")
        assert allowed is False
        assert "correlated" in reason.lower()

    def test_correlation_filter_allows(self, risk_manager):
        """Should allow when correlation is low."""
        risk_manager.state["open_positions"] = {
            "BTC/USDT": {"size_usdt": 100},
        }
        risk_manager.correlation_filter.can_open_position.return_value = (
            True, "OK"
        )

        allowed, reason = risk_manager.can_trade(symbol="DOGE/USDT")
        assert allowed is True


class TestPositionManagement:
    """Test opening/closing positions."""

    def test_open_position_deducts_capital(self, risk_manager):
        """Opening a position should deduct from capital."""
        initial_capital = risk_manager.state["capital"]

        risk_manager.open_position(
            symbol="BTC/USDT",
            side="long",
            entry_price=50000,
            stop_loss_pct=0.03,
        )

        assert risk_manager.state["capital"] < initial_capital
        assert "BTC/USDT" in risk_manager.state["open_positions"]

    def test_close_profitable_position(self, risk_manager):
        """Closing a profitable long should increase capital."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)
        capital_after_open = risk_manager.state["capital"]

        trade = risk_manager.close_position("BTC/USDT", 52000, reason="signal")

        assert trade is not None
        assert trade["pnl_usd"] > 0
        assert risk_manager.state["capital"] > capital_after_open
        assert "BTC/USDT" not in risk_manager.state["open_positions"]

    def test_close_losing_position(self, risk_manager):
        """Closing a losing long should decrease capital vs entry."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)

        trade = risk_manager.close_position("BTC/USDT", 48000, reason="stop_loss")

        assert trade is not None
        assert trade["pnl_usd"] < 0
        assert risk_manager.state["consecutive_losses"] == 1

    def test_consecutive_loss_tracking(self, risk_manager):
        """Consecutive losses should be tracked correctly."""
        # Trade 1: loss
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)
        risk_manager.close_position("BTC/USDT", 48000)
        assert risk_manager.state["consecutive_losses"] == 1

        # Trade 2: loss
        risk_manager.open_position("ETH/USDT", "long", 3000, 0.03)
        risk_manager.close_position("ETH/USDT", 2800)
        assert risk_manager.state["consecutive_losses"] == 2

        # Trade 3: win — should reset
        risk_manager.open_position("SOL/USDT", "long", 100, 0.03)
        risk_manager.close_position("SOL/USDT", 110)
        assert risk_manager.state["consecutive_losses"] == 0

    def test_close_nonexistent_position(self, risk_manager):
        """Closing a position that doesn't exist should return None."""
        result = risk_manager.close_position("FAKE/USDT", 100)
        assert result is None

    def test_stop_loss_check(self, risk_manager):
        """check_stops_and_tp should close positions that hit stop."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)

        # Price dropped below stop (50000 * 0.97 = 48500)
        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 48000})

        assert len(closed) == 1
        assert closed[0]["reason"] == "stop_loss"

    def test_stop_loss_not_triggered(self, risk_manager):
        """check_stops_and_tp should not close if price is above stop."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)

        closed = risk_manager.check_stops_and_tp({"BTC/USDT": 49000})

        assert len(closed) == 0

    def test_state_persistence(self, risk_manager, temp_state_file):
        """State should be saved to file after operations."""
        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)

        assert temp_state_file.exists()
        with open(temp_state_file) as f:
            saved = json.load(f)
        assert "BTC/USDT" in saved["open_positions"]

    def test_peak_capital_updates(self, risk_manager):
        """Peak capital should update when capital increases."""
        initial_peak = risk_manager.state["peak_capital"]

        risk_manager.open_position("BTC/USDT", "long", 50000, 0.03)
        risk_manager.close_position("BTC/USDT", 55000)

        assert risk_manager.state["peak_capital"] >= initial_peak
