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
            "initial_capital": INITIAL_CAPITAL,
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
        # Cash + locked equity per open position. For futures, only margin was deducted
        # from capital, so equity contribution is margin (NOT notional size_usdt — using
        # notional would inflate equity by leverage and hide the real drawdown).
        total = self.state["capital"]
        for pos in self.state["open_positions"].values():
            if pos.get("leverage") and pos.get("margin"):
                total += pos["margin"]
            else:
                total += pos["size_usdt"]
        if peak <= 0:
            return 0
        return (total - peak) / peak

    # ── Position Sizing ──

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly criterion fraction based on recent trade history.

        Kelly f* = (W * R - L) / R
        where:
          W = win rate
          R = avg_win / avg_loss ratio
          L = loss rate (1 - W)

        Uses last 20 trades for calculation.
        Returns half-Kelly (more conservative) clamped to [0.5, 2.0] as a multiplier.
        If not enough trades (<5), returns 1.0 (no adjustment).
        """
        history = self.state.get("trade_history", [])
        recent = history[-20:] if len(history) >= 5 else None

        if recent is None:
            return 1.0

        wins = [t for t in recent if t["pnl_usd"] > 0]
        losses = [t for t in recent if t["pnl_usd"] <= 0]

        if not wins or not losses:
            return 1.0

        win_rate = len(wins) / len(recent)
        avg_win = sum(t["pnl_usd"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl_usd"] for t in losses) / len(losses))

        if avg_loss == 0:
            return 1.0

        R = avg_win / avg_loss
        kelly = (win_rate * R - (1 - win_rate)) / R

        # Half-Kelly for safety
        half_kelly = kelly / 2

        # Convert to multiplier: if kelly suggests 4% and base is 2%, multiplier = 2.0
        # Clamp between 0.5x and 2.0x of base risk
        multiplier = max(0.5, min(2.0, 1.0 + half_kelly))

        return round(multiplier, 3)

    def _recent_performance_multiplier(self) -> float:
        """
        Scale position size based on recent performance (last 10 trades).

        - If recent Sharpe-like metric is positive (more wins, good R:R): scale up to 1.3x
        - If negative (losing streak approaching): scale down to 0.7x
        - Neutral: 1.0x

        This is separate from the consecutive_losses reduction (which is a hard cutoff at 3).
        """
        history = self.state.get("trade_history", [])
        recent = history[-10:] if len(history) >= 5 else None

        if recent is None:
            return 1.0

        pnls = [t["pnl_usd"] for t in recent]
        avg_pnl = sum(pnls) / len(pnls)

        if avg_pnl > 0:
            # Winning: scale up slightly (max 1.3x)
            multiplier = min(1.3, 1.0 + avg_pnl / 50)  # +$50 avg -> 2.0x, capped at 1.3
        else:
            # Losing: scale down (min 0.7x)
            multiplier = max(0.7, 1.0 + avg_pnl / 50)  # -$50 avg -> 0.0x, capped at 0.7

        return round(multiplier, 3)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float,
        side: str = "long",
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

        # Kelly criterion adjustment
        kelly_mult = self._calculate_kelly_fraction()
        risk_amount *= kelly_mult

        # Recent performance adjustment
        perf_mult = self._recent_performance_multiplier()
        risk_amount *= perf_mult

        if kelly_mult != 1.0 or perf_mult != 1.0:
            logger.info(f"  Position sizing: kelly={kelly_mult:.2f}x, perf={perf_mult:.2f}x")

        # Position size = risk_amount / stop_loss_distance
        stop_distance = entry_price * stop_loss_pct
        position_size_base = risk_amount / stop_distance  # In base asset units
        position_value = position_size_base * entry_price  # In USDT

        # Cap at available capital (no leverage)
        max_position = capital * 0.4  # Max 40% of capital per position
        if position_value > max_position:
            position_value = max_position
            position_size_base = position_value / entry_price

        if side == "short":
            stop_price = entry_price * (1 + stop_loss_pct)
        else:
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

    def adjust_stop_loss(self, base_stop_pct: float, regime: str, atr_pct: float = None) -> float:
        """
        Adjust stop-loss based on market regime and volatility.

        - Trending: widen stop by 1.3x (avoid premature exit in trends)
        - Sideways: keep base stop (mean-reversion works with tighter stops)
        - Choppy: tighten stop by 0.8x (cut losses faster in noise)

        Also clamp based on ATR if available:
        - Stop should be at least 1x ATR% (avoid getting stopped by normal noise)
        - Stop should be at most 3x ATR% (don't hold losers too long)

        Final clamp: [1.5%, 8%] regardless
        """
        # Regime multiplier
        multipliers = {
            "trending": 1.3,
            "sideways": 1.0,
            "choppy": 0.8,
        }
        mult = multipliers.get(regime, 1.0)
        adjusted = base_stop_pct * mult

        # ATR-based bounds if available
        if atr_pct and atr_pct > 0:
            atr_frac = atr_pct / 100
            min_stop = atr_frac * 1.0  # At least 1x ATR
            max_stop = atr_frac * 3.0  # At most 3x ATR
            adjusted = max(adjusted, min_stop)
            adjusted = min(adjusted, max_stop)

        # Hard clamps
        adjusted = max(adjusted, 0.015)  # minimum 1.5%
        adjusted = min(adjusted, 0.08)   # maximum 8%

        return round(adjusted, 4)

    def open_position(self, symbol: str, side: str, entry_price: float, stop_loss_pct: float, atr_pct: float = None, regime: str = None, strategy: str = None) -> dict:
        """Open a paper position."""
        # Adjust stop based on regime
        if regime:
            actual_stop_pct = self.adjust_stop_loss(stop_loss_pct, regime, atr_pct)
            logger.info(f"  Stop adjusted: {stop_loss_pct:.1%} -> {actual_stop_pct:.1%} (regime={regime})")
        else:
            actual_stop_pct = stop_loss_pct

        sizing = self.calculate_position_size(symbol, entry_price, actual_stop_pct, side=side)

        # Take-profit levels based on risk distance
        risk_distance = entry_price * actual_stop_pct
        if side == "short":
            tp1_price = round(entry_price - risk_distance * 2, 4)  # 2R
            tp2_price = round(entry_price - risk_distance * 3, 4)  # 3R
        else:
            tp1_price = round(entry_price + risk_distance * 2, 4)  # 2R
            tp2_price = round(entry_price + risk_distance * 3, 4)  # 3R

        position = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size_base": sizing["size_base"],
            "size_usdt": sizing["size_usdt"],
            "stop_price": sizing["stop_price"],
            "stop_loss_pct": actual_stop_pct,
            "base_stop_loss_pct": stop_loss_pct,
            "regime": regime,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp1_hit": False,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "atr_pct": atr_pct,
            "strategy": strategy or "unknown",
        }

        self.state["open_positions"][symbol] = position

        # Deduct position value + entry fee from capital. Exit fee is taken at
        # close_position(); without entry fee here, capital drifted positive
        # vs realized PnL accumulator.
        entry_fee = sizing["size_usdt"] * TRADING_FEE
        self.state["capital"] -= sizing["size_usdt"] + entry_fee
        self.state["total_pnl"] -= entry_fee

        self._save_state()

        logger.info(
            f"OPENED {side.upper()} {symbol}: "
            f"size=${sizing['size_usdt']:.2f} ({sizing['capital_pct']:.1f}% capital), "
            f"entry=${entry_price:.4f}, stop=${sizing['stop_price']:.4f}"
        )

        return position

    def _partial_close(self, symbol: str, pct: float, exit_price: float, reason: str) -> dict:
        """Close a fraction of an open position.

        Args:
            symbol: Trading pair symbol.
            pct: Fraction to close (0.0 - 1.0).
            exit_price: Current market price for the partial exit.
            reason: Why the partial close is happening.

        Returns:
            A trade_record dict for the partial close.
        """
        pos = self.state["open_positions"][symbol]
        entry = pos["entry_price"]
        partial_size = pos["size_base"] * pct
        partial_value = pos["size_usdt"] * pct

        # Calculate partial PnL
        if pos["side"] == "long":
            partial_pnl = partial_size * (exit_price - entry)
        else:
            partial_pnl = partial_size * (entry - exit_price)

        # Deduct exit fee on the partial value
        exit_fee = partial_value * TRADING_FEE
        partial_pnl -= exit_fee

        # Return partial value + PnL to capital
        self.state["capital"] += partial_value + partial_pnl

        # Reduce position size
        pos["size_base"] = round(pos["size_base"] * (1 - pct), 6)
        pos["size_usdt"] = round(pos["size_usdt"] * (1 - pct), 2)

        self._save_state()

        pnl_pct = ((exit_price - entry) / entry) if pos["side"] == "long" else ((entry - exit_price) / entry)

        trade_record = {
            "symbol": symbol,
            "side": pos["side"],
            "entry_price": entry,
            "exit_price": exit_price,
            "partial_pct": pct,
            "size_base": round(partial_size, 6),
            "size_usdt": round(partial_value, 2),
            "pnl_pct": round(pnl_pct * 100, 3),
            "pnl_usd": round(partial_pnl, 2),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"PARTIAL CLOSE {symbol} ({pct*100:.0f}%): {reason}, "
            f"PnL=${partial_pnl:+.2f}, "
            f"Remaining size={pos['size_base']}"
        )

        return trade_record

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
        closed_at = datetime.now(timezone.utc)
        duration_hours = None
        if pos.get("opened_at"):
            try:
                opened_at = datetime.fromisoformat(pos["opened_at"])
                duration_hours = round((closed_at - opened_at).total_seconds() / 3600, 2)
            except (ValueError, TypeError):
                pass

        trade_record = {
            **pos,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct * 100, 3),
            "pnl_usd": round(pnl_usd, 2),
            "reason": reason,
            "closed_at": closed_at.isoformat(),
            "duration_hours": duration_hours,
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

    # Trailing stop config
    BREAKEVEN_THRESHOLD = 0.015   # Move stop to breakeven when profit >= 1.5%
    TRAIL_ACTIVATE_THRESHOLD = 0.025  # Start trailing when profit >= 2.5%
    TRAIL_DISTANCE_PCT = 0.015    # Trail 1.5% behind highest profit

    def check_stops_and_tp(self, current_prices: dict) -> list:
        """Check stop-losses, take-profit levels, and trailing stops.

        Trailing stop logic (3 phases):
          1. profit < 1.5%  → fixed stop loss (original)
          2. profit >= 1.5% → move stop to breakeven (entry price)
          3. profit >= 2.5% → trail stop behind best price (1.5% distance,
             or 2x ATR if available). Stop only moves up, never down.
        After TP1 hit, trailing uses tighter distance (ATR-based).
        """
        closed = []
        state_changed = False

        for symbol, pos in list(self.state["open_positions"].items()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            side = pos["side"]
            entry = pos["entry_price"]
            tp1_hit = pos.get("tp1_hit", False)
            tp1_price = pos.get("tp1_price")
            tp2_price = pos.get("tp2_price")

            # Update highest/lowest price tracking
            if side == "long":
                if price > pos.get("highest_price", entry):
                    pos["highest_price"] = price
                    state_changed = True
            else:
                if price < pos.get("lowest_price", entry):
                    pos["lowest_price"] = price
                    state_changed = True

            # Calculate unrealized profit %
            if side == "long":
                unrealized_pct = (price - entry) / entry
                best_price = pos.get("highest_price", entry)
            else:
                unrealized_pct = (entry - price) / entry
                best_price = pos.get("lowest_price", entry)

            # ── Take-Profit 2 (3R) ── close remaining 100%
            if tp1_hit and tp2_price is not None:
                if (side == "long" and price >= tp2_price) or (side == "short" and price <= tp2_price):
                    trade = self.close_position(symbol, price, reason="take_profit_2")
                    if trade:
                        closed.append(trade)
                    continue

            # ── Take-Profit 1 (2R) ── partial close 50%
            if not tp1_hit and tp1_price is not None:
                if (side == "long" and price >= tp1_price) or (side == "short" and price <= tp1_price):
                    trade = self._partial_close(symbol, 0.5, price, "take_profit_1")
                    if trade:
                        closed.append(trade)
                    pos["tp1_hit"] = True
                    pos["stop_price"] = entry  # breakeven
                    state_changed = True
                    continue

            # ── Trailing Stop Logic ──
            trail_pct = self._get_trail_distance(pos, tp1_hit)

            if tp1_hit or unrealized_pct >= self.TRAIL_ACTIVATE_THRESHOLD:
                # Phase 3: Active trailing — stop follows best price
                if side == "long":
                    new_stop = best_price * (1 - trail_pct)
                    if new_stop > pos["stop_price"]:
                        pos["stop_price"] = round(new_stop, 4)
                        state_changed = True
                        logger.debug(f"Trailing stop updated {symbol}: stop=${new_stop:.4f} (best=${best_price:.4f})")
                else:
                    new_stop = best_price * (1 + trail_pct)
                    if new_stop < pos["stop_price"]:
                        pos["stop_price"] = round(new_stop, 4)
                        state_changed = True
                        logger.debug(f"Trailing stop updated {symbol}: stop=${new_stop:.4f} (best=${best_price:.4f})")

            elif unrealized_pct >= self.BREAKEVEN_THRESHOLD:
                # Phase 2: Move stop to breakeven
                if side == "long" and pos["stop_price"] < entry:
                    pos["stop_price"] = entry
                    state_changed = True
                    logger.info(f"Breakeven stop activated {symbol}: stop moved to entry ${entry:.4f}")
                elif side == "short" and pos["stop_price"] > entry:
                    pos["stop_price"] = entry
                    state_changed = True
                    logger.info(f"Breakeven stop activated {symbol}: stop moved to entry ${entry:.4f}")

            # ── Check if stop price is hit ──
            # Label as trailing_stop only if stop has actually moved closer to price
            # than the initial stop by >1% buffer. Old code used *1.01 which made the
            # threshold looser than the initial stop, so every untouched stop was
            # mislabeled as trailing_stop.
            if side == "long" and price <= pos["stop_price"]:
                reason = "trailing_stop" if pos["stop_price"] > entry * (1 - pos["stop_loss_pct"] * 0.99) else "stop_loss"
                trade = self.close_position(symbol, price, reason=reason)
                if trade:
                    closed.append(trade)
            elif side == "short" and price >= pos["stop_price"]:
                reason = "trailing_stop" if pos["stop_price"] < entry * (1 + pos["stop_loss_pct"] * 0.99) else "stop_loss"
                trade = self.close_position(symbol, price, reason=reason)
                if trade:
                    closed.append(trade)

        if state_changed:
            self._save_state()

        return closed

    def _get_trail_distance(self, pos, tp1_hit):
        """Calculate trailing distance percentage based on ATR or default."""
        atr_pct = pos.get("atr_pct")
        if tp1_hit and atr_pct:
            # After TP1: tighter trail using 2x ATR
            trail_pct = (atr_pct / 100) * 2
            trail_pct = max(trail_pct, 0.01)   # min 1%
            trail_pct = min(trail_pct, 0.08)   # max 8%
        elif atr_pct:
            # Before TP1: wider trail using 2.5x ATR
            trail_pct = (atr_pct / 100) * 2.5
            trail_pct = max(trail_pct, self.TRAIL_DISTANCE_PCT)
            trail_pct = min(trail_pct, 0.10)
        else:
            trail_pct = self.TRAIL_DISTANCE_PCT
        return trail_pct

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
