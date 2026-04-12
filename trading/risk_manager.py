"""
Risk Manager
==============
Manages position sizing, stop-loss, and kill switch conditions.
Enforces all rules from the risk management framework.

Rules:
  - Max risk/trade: 2% of capital
  - Max positions: 3 simultaneous
  - Max drawdown: 15% → kill switch
  - Stop-loss: 3% (from optimization)
  - 3 consecutive losses → reduce size 50%
  - 5 consecutive losses → stop trading 1 week
"""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

from config.settings import (
    INITIAL_CAPITAL,
    MAX_RISK_PER_TRADE,
    MAX_POSITIONS,
    MAX_DRAWDOWN,
    TRADING_FEE,
)
from utils.correlation import CorrelationFilter


STATE_FILE = Path("trading/state.json")


class RiskManager:
    """Manages risk for paper/live trading."""

    def __init__(self):
        self.state = self._load_state()
        self.correlation_filter = CorrelationFilter()
        logger.info(
            f"RiskManager loaded. Capital: ${self.state['capital']:.2f}, "
            f"Positions: {len(self.state['open_positions'])}"
        )

    def _default_state(self) -> dict:
        return {
            "capital": INITIAL_CAPITAL,
            "peak_capital": INITIAL_CAPITAL,
            "open_positions": {},
            "trade_history": [],
            "consecutive_losses": 0,
            "total_trades": 0,
            "total_wins": 0,
            "total_pnl": 0.0,
            "paused_until": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return self._default_state()

    def _save_state(self):
        STATE_FILE.parent.mkdir(exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    # ── Risk Checks ──

    def can_trade(self, symbol: str = None) -> tuple:
        """
        Check if trading is allowed.

        Args:
            symbol: Symbol to check correlation against open positions.
                    If None, only checks general risk rules.

        Returns (allowed: bool, reason: str)
        """
        # Check pause
        if self.state["paused_until"]:
            pause_end = datetime.fromisoformat(self.state["paused_until"])
            if datetime.now(timezone.utc) < pause_end:
                remaining = (pause_end - datetime.now(timezone.utc)).days
                return False, f"Trading paused. Resumes in {remaining} days"
            else:
                self.state["paused_until"] = None
                self.state["consecutive_losses"] = 0
                self._save_state()

        # Kill switch: max drawdown
        drawdown = self._current_drawdown()
        if drawdown < -MAX_DRAWDOWN:
            return False, (
                f"KILL SWITCH: Drawdown {drawdown:.1%} exceeds "
                f"max {MAX_DRAWDOWN:.0%}. Stop trading, review system."
            )

        # Max positions
        if len(self.state["open_positions"]) >= MAX_POSITIONS:
            return False, f"Max positions reached ({MAX_POSITIONS})"

        # 5 consecutive losses → pause 1 week
        if self.state["consecutive_losses"] >= 5:
            pause_until = datetime.now(timezone.utc) + timedelta(days=7)
            self.state["paused_until"] = pause_until.isoformat()
            self._save_state()
            return False, "5 consecutive losses. Pausing for 1 week."

        # Correlation filter: block if too correlated with open positions
        if symbol and self.state["open_positions"]:
            open_symbols = list(self.state["open_positions"].keys())
            allowed, reason = self.correlation_filter.can_open_position(
                symbol, open_symbols
            )
            if not allowed:
                return False, reason

        return True, "OK"

    def _current_drawdown(self) -> float:
        peak = self.state["peak_capital"]
        # Total equity = cash + open position values
        total = self.state["capital"]
        for pos in self.state["open_positions"].values():
            total += pos["size_usdt"]
        if peak <= 0:
            return 0
        return (total - peak) / peak

    # ── Position Sizing ──

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float,
    ) -> dict:
        """
        Calculate position size based on risk management rules.

        Returns:
            dict with size, risk_amount, stop_price, etc.
        """
        capital = self.state["capital"]

        # Risk amount = capital * max_risk_per_trade
        risk_amount = capital * MAX_RISK_PER_TRADE

        # Reduce size after 3 consecutive losses
        if self.state["consecutive_losses"] >= 3:
            risk_amount *= 0.5
            logger.warning("3+ consecutive losses: position size reduced 50%")

        # Position size = risk_amount / stop_loss_distance
        stop_distance = entry_price * stop_loss_pct
        position_size_base = risk_amount / stop_distance  # In base asset units
        position_value = position_size_base * entry_price  # In USDT

        # Cap at available capital (no leverage)
        max_position = capital * 0.4  # Max 40% of capital per position
        if position_value > max_position:
            position_value = max_position
            position_size_base = position_value / entry_price

        stop_price = entry_price * (1 - stop_loss_pct)

        # Account for fees
        fee_cost = position_value * TRADING_FEE * 2  # Entry + exit

        return {
            "symbol": symbol,
            "size_base": round(position_size_base, 6),
            "size_usdt": round(position_value, 2),
            "entry_price": entry_price,
            "stop_price": round(stop_price, 4),
            "stop_loss_pct": stop_loss_pct,
            "risk_amount": round(risk_amount, 2),
            "estimated_fee": round(fee_cost, 2),
            "capital_pct": round(position_value / capital * 100, 1),
        }

    # ── Position Management ──

    def open_position(self, symbol: str, side: str, entry_price: float, stop_loss_pct: float) -> dict:
        """Open a paper position."""
        sizing = self.calculate_position_size(symbol, entry_price, stop_loss_pct)

        position = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size_base": sizing["size_base"],
            "size_usdt": sizing["size_usdt"],
            "stop_price": sizing["stop_price"],
            "stop_loss_pct": stop_loss_pct,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }

        self.state["open_positions"][symbol] = position

        # Deduct position value + entry fee from capital
        self.state["capital"] -= sizing["size_usdt"]
        
        self._save_state()

        logger.info(
            f"OPENED {side.upper()} {symbol}: "
            f"size=${sizing['size_usdt']:.2f} ({sizing['capital_pct']:.1f}% capital), "
            f"entry=${entry_price:.4f}, stop=${sizing['stop_price']:.4f}"
        )

        return position

    def close_position(self, symbol: str, exit_price: float, reason: str = "signal") -> dict:
        """Close a paper position and record PnL."""
        if symbol not in self.state["open_positions"]:
            logger.warning(f"No open position for {symbol}")
            return None

        pos = self.state["open_positions"][symbol]
        entry = pos["entry_price"]
        size_base = pos["size_base"]
        side = pos["side"]

        # Calculate PnL
        if side == "long":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl_usd = size_base * abs(exit_price - entry)
        if pnl_pct < 0:
            pnl_usd = -pnl_usd

        # Deduct exit fee
        exit_fee = pos["size_usdt"] * TRADING_FEE
        pnl_usd -= exit_fee

        # Update capital
        self.state["capital"] += pos["size_usdt"] + pnl_usd

        # Update peak
        if self.state["capital"] > self.state["peak_capital"]:
            self.state["peak_capital"] = self.state["capital"]

        # Track wins/losses
        self.state["total_trades"] += 1
        self.state["total_pnl"] += pnl_usd

        if pnl_usd > 0:
            self.state["total_wins"] += 1
            self.state["consecutive_losses"] = 0
        else:
            self.state["consecutive_losses"] += 1

        # Record trade
        trade_record = {
            **pos,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct * 100, 3),
            "pnl_usd": round(pnl_usd, 2),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["trade_history"].append(trade_record)

        # Remove from open
        del self.state["open_positions"][symbol]

        self._save_state()

        emoji = "✅" if pnl_usd > 0 else "❌"
        logger.info(
            f"{emoji} CLOSED {symbol}: {reason}, "
            f"PnL={pnl_pct*100:+.2f}% (${pnl_usd:+.2f}), "
            f"Capital=${self.state['capital']:.2f}"
        )

        return trade_record

    def check_stop_losses(self, current_prices: dict) -> list:
        """Check if any open positions hit stop-loss."""
        closed = []
        for symbol, pos in list(self.state["open_positions"].items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            if pos["side"] == "long" and price <= pos["stop_price"]:
                trade = self.close_position(symbol, price, reason="stop_loss")
                if trade:
                    closed.append(trade)
            elif pos["side"] == "short" and price >= pos["stop_price"] * (2 - 1):
                # Short stop = entry * (1 + stop_loss_pct)
                short_stop = pos["entry_price"] * (1 + pos["stop_loss_pct"])
                if price >= short_stop:
                    trade = self.close_position(symbol, price, reason="stop_loss")
                    if trade:
                        closed.append(trade)

        return closed

    # ── Reporting ──

    def get_summary(self) -> str:
        """Get formatted summary of current state."""
        s = self.state
        drawdown = self._current_drawdown()
        total_equity = s["capital"]
        for pos in s["open_positions"].values():
            total_equity += pos["size_usdt"]
        win_rate = (s["total_wins"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0

        lines = [
            "=" * 50,
            "  PAPER TRADING STATUS",
            "=" * 50,
            f"  Cash:             ${s['capital']:.2f}",
            f"  Open Positions:   ${total_equity - s['capital']:.2f}",
            f"  Total Equity:     ${total_equity:.2f}",
            f"  Peak:             ${s['peak_capital']:.2f}",
            f"  Drawdown:         {drawdown:.2%}",
            f"  Total PnL:        ${s['total_pnl']:+.2f}",
        ]

        for sym, pos in s["open_positions"].items():
            lines.append(f"    {sym}: {pos['side']} @ ${pos['entry_price']:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def reset(self):
        """Reset all state (use with caution)."""
        self.state = self._default_state()
        self._save_state()
        logger.info("RiskManager state RESET.")
