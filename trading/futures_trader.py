"""
Futures Paper Trading Bot
==========================
Extends paper trading to support Binance USDT-M Futures.

Features:
- Long AND short positions (real short, not just exit)
- Leverage support (default 3x, max configurable)
- Isolated margin mode
- Funding rate tracking
- Same risk management (stop-loss, trailing stop, take-profit)

Usage:
    # Paper futures trading
    python -m trading.futures_trader --run --interval 0.5

    # Check status
    python -m trading.futures_trader --status
"""

import argparse
import json
import time
import os
import sys
sys.path.insert(0, ".")

import ccxt
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from config.settings import (
    LOG_FILE,
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    FUTURES_LEVERAGE,
    FUTURES_FEE,
    FUTURES_MARGIN_TYPE,
    INITIAL_CAPITAL,
    MAX_RISK_PER_TRADE,
    MAX_POSITIONS,
    MAX_DRAWDOWN,
    VN_TIMEZONE,
)
from trading.signal_generator import SignalGenerator, ALPHA_CONFIGS, FUTURES_ALPHA_CONFIGS
from trading.risk_manager import RiskManager
from trading import manual_actions
from strategies.onchain_alphas import OnchainSignalFilter
from utils.telegram import TelegramAlert
from data.fetcher import BinanceFetcher
from data.models import init_db

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

FUTURES_STATE_FILE = Path("trading/futures_state.json")


class FuturesRiskManager(RiskManager):
    """Extended risk manager for futures trading with leverage."""

    def __init__(self):
        self.state_file = FUTURES_STATE_FILE
        self.state = self._load_futures_state()
        from utils.correlation import CorrelationFilter
        self.correlation_filter = CorrelationFilter()
        logger.info(
            f"FuturesRiskManager loaded. Capital: ${self.state['capital']:.2f}, "
            f"Leverage: {FUTURES_LEVERAGE}x, "
            f"Positions: {len(self.state['open_positions'])}"
        )

    def _load_futures_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load futures state: {e}")
        return self._default_state()

    def _save_state(self):
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def calculate_position_size(self, symbol, entry_price, stop_loss_pct, side="long"):
        """Calculate position size with leverage."""
        capital = self.state["capital"]
        risk_amount = capital * MAX_RISK_PER_TRADE

        if self.state["consecutive_losses"] >= 3:
            risk_amount *= 0.5
            logger.warning("3+ consecutive losses: position size reduced 50%")

        # With leverage, we can control larger position with same margin
        stop_distance = entry_price * stop_loss_pct
        position_size_base = risk_amount / stop_distance
        position_value = position_size_base * entry_price  # Notional value

        # Margin required = notional / leverage
        margin_required = position_value / FUTURES_LEVERAGE

        # Cap margin at 40% of capital
        max_margin = capital * 0.4
        if margin_required > max_margin:
            margin_required = max_margin
            position_value = margin_required * FUTURES_LEVERAGE
            position_size_base = position_value / entry_price

        if side == "short":
            stop_price = entry_price * (1 + stop_loss_pct)
        else:
            stop_price = entry_price * (1 - stop_loss_pct)
        fee_cost = position_value * FUTURES_FEE * 2

        return {
            "symbol": symbol,
            "size_base": round(position_size_base, 6),
            "size_usdt": round(position_value, 2),  # Notional
            "margin": round(margin_required, 2),
            "leverage": FUTURES_LEVERAGE,
            "entry_price": entry_price,
            "stop_price": round(stop_price, 4),
            "stop_loss_pct": stop_loss_pct,
            "risk_amount": round(risk_amount, 2),
            "estimated_fee": round(fee_cost, 2),
            "capital_pct": round(margin_required / capital * 100, 1),
        }

    def open_position(self, symbol, side, entry_price, stop_loss_pct, atr_pct=None, regime=None, strategy=None):
        """Open a futures position (long or short)."""
        # Adjust stop based on regime
        if regime:
            actual_stop_pct = self.adjust_stop_loss(stop_loss_pct, regime, atr_pct)
            logger.info(f"  Stop adjusted: {stop_loss_pct:.1%} -> {actual_stop_pct:.1%} (regime={regime})")
        else:
            actual_stop_pct = stop_loss_pct

        sizing = self.calculate_position_size(symbol, entry_price, actual_stop_pct, side=side)

        risk_distance = entry_price * actual_stop_pct
        if side == "long":
            tp1_price = round(entry_price + risk_distance * 2, 4)
            tp2_price = round(entry_price + risk_distance * 3, 4)
        else:
            tp1_price = round(entry_price - risk_distance * 2, 4)
            tp2_price = round(entry_price - risk_distance * 3, 4)

        # Liquidation price (approximate for isolated margin)
        if side == "long":
            liq_price = round(entry_price * (1 - 1/FUTURES_LEVERAGE + 0.005), 4)
        else:
            liq_price = round(entry_price * (1 + 1/FUTURES_LEVERAGE - 0.005), 4)

        position = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "size_base": sizing["size_base"],
            "size_usdt": sizing["size_usdt"],  # Notional
            "margin": sizing["margin"],
            "leverage": FUTURES_LEVERAGE,
            "stop_price": sizing["stop_price"],
            "stop_loss_pct": actual_stop_pct,
            "base_stop_loss_pct": stop_loss_pct,
            "regime": regime,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp1_hit": False,
            "liq_price": liq_price,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "atr_pct": atr_pct,
            "funding_paid": 0.0,
            "strategy": strategy or "unknown",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "mode": "futures",
        }

        self.state["open_positions"][symbol] = position

        # Deduct margin (not full notional) from capital
        self.state["capital"] -= sizing["margin"]

        self._save_state()

        logger.info(
            f"OPENED {side.upper()} {symbol} (Futures {FUTURES_LEVERAGE}x): "
            f"notional=${sizing['size_usdt']:.2f}, margin=${sizing['margin']:.2f} "
            f"({sizing['capital_pct']:.1f}% capital), "
            f"entry=${entry_price:.4f}, stop=${sizing['stop_price']:.4f}, "
            f"liq=${liq_price:.4f}, TP1=${tp1_price:.4f}, TP2=${tp2_price:.4f}"
        )

        return position

    def _partial_close(self, symbol, pct, exit_price, reason):
        """Close a fraction of a futures position. Margin-aware (capital was
        decremented by margin only at entry, so we return margin*pct + PnL)."""
        if symbol not in self.state["open_positions"]:
            return None
        pct = max(0.01, min(1.0, float(pct)))
        pos = self.state["open_positions"][symbol]
        entry = pos["entry_price"]
        side = pos["side"]

        partial_base = pos["size_base"] * pct
        partial_notional = pos["size_usdt"] * pct
        partial_margin = pos.get("margin", pos["size_usdt"] / FUTURES_LEVERAGE) * pct

        if side == "long":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry
        pnl_usd = partial_base * (exit_price - entry)
        if side == "short":
            pnl_usd = -pnl_usd
        pnl_usd -= partial_notional * FUTURES_FEE
        pnl_usd -= pos.get("funding_paid", 0) * pct

        self.state["capital"] += partial_margin + pnl_usd
        if self.state["capital"] > self.state["peak_capital"]:
            self.state["peak_capital"] = self.state["capital"]
        self.state["total_pnl"] += pnl_usd

        pos["size_base"] = round(pos["size_base"] - partial_base, 6)
        pos["size_usdt"] = round(pos["size_usdt"] - partial_notional, 2)
        pos["margin"] = round(pos.get("margin", 0) - partial_margin, 2)
        pos["funding_paid"] = pos.get("funding_paid", 0) * (1 - pct)

        roe = (pnl_usd / partial_margin * 100) if partial_margin > 0 else 0
        trade_record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "partial_pct": round(pct, 3),
            "size_base": round(partial_base, 6),
            "size_usdt": round(partial_notional, 2),
            "margin": round(partial_margin, 2),
            "pnl_pct": round(pnl_pct * 100, 3),
            "pnl_usd": round(pnl_usd, 2),
            "roe": round(roe, 2),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "mode": "futures",
        }
        self.state["trade_history"].append(trade_record)
        self._save_state()
        logger.info(
            f"PARTIAL CLOSE {symbol} Futures ({pct*100:.0f}%): {reason}, "
            f"PnL=${pnl_usd:+.2f}, ROE={roe:+.1f}%"
        )
        return trade_record

    def close_position(self, symbol, exit_price, reason="signal"):
        """Close a futures position and compute leveraged PnL."""
        if symbol not in self.state["open_positions"]:
            logger.warning(f"No open position for {symbol}")
            return None

        pos = self.state["open_positions"][symbol]
        entry = pos["entry_price"]
        size_base = pos["size_base"]
        side = pos["side"]
        margin = pos.get("margin", pos["size_usdt"] / FUTURES_LEVERAGE)

        # PnL on notional
        if side == "long":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl_usd = size_base * abs(exit_price - entry)
        if pnl_pct < 0:
            pnl_usd = -pnl_usd

        # Deduct fees + funding
        exit_fee = pos["size_usdt"] * FUTURES_FEE
        pnl_usd -= exit_fee
        pnl_usd -= pos.get("funding_paid", 0)

        # Return margin + PnL
        self.state["capital"] += margin + pnl_usd

        if self.state["capital"] > self.state["peak_capital"]:
            self.state["peak_capital"] = self.state["capital"]

        self.state["total_trades"] += 1
        self.state["total_pnl"] += pnl_usd

        if pnl_usd > 0:
            self.state["total_wins"] += 1
            self.state["consecutive_losses"] = 0
        else:
            self.state["consecutive_losses"] += 1

        # Leveraged return on margin
        roe = pnl_usd / margin * 100 if margin > 0 else 0

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
            "roe": round(roe, 2),  # Return on Equity (leveraged)
            "duration_hours": duration_hours,
            "reason": reason,
            "closed_at": closed_at.isoformat(),
        }
        self.state["trade_history"].append(trade_record)
        del self.state["open_positions"][symbol]
        self._save_state()

        tag = "[WIN]" if pnl_usd > 0 else "[LOSS]"
        logger.info(
            f"{tag} CLOSED {symbol} (Futures): {reason}, "
            f"PnL={pnl_pct*100:+.2f}% (${pnl_usd:+.2f}), "
            f"ROE={roe:+.1f}%, Capital=${self.state['capital']:.2f}"
        )

        return trade_record

    def get_summary(self) -> str:
        s = self.state
        drawdown = self._current_drawdown()
        total_equity = s["capital"]
        margin_used = 0
        for pos in s["open_positions"].values():
            margin_used += pos.get("margin", pos["size_usdt"] / FUTURES_LEVERAGE)
            total_equity += pos.get("margin", pos["size_usdt"] / FUTURES_LEVERAGE)
        win_rate = (s["total_wins"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0

        lines = [
            "=" * 55,
            "  FUTURES PAPER TRADING STATUS",
            f"  Leverage: {FUTURES_LEVERAGE}x | Margin: {FUTURES_MARGIN_TYPE}",
            "=" * 55,
            f"  Cash:             ${s['capital']:.2f}",
            f"  Margin Used:      ${margin_used:.2f}",
            f"  Total Equity:     ${total_equity:.2f}",
            f"  Peak:             ${s['peak_capital']:.2f}",
            f"  Drawdown:         {drawdown:.2%}",
            f"  Total PnL:        ${s['total_pnl']:+.2f}",
            f"  Trades: {s['total_trades']} | Win Rate: {win_rate:.0f}%",
        ]

        for sym, pos in s["open_positions"].items():
            lev = pos.get("leverage", FUTURES_LEVERAGE)
            lines.append(
                f"    {sym}: {pos['side'].upper()} {lev}x "
                f"@ ${pos['entry_price']:.4f} | "
                f"Margin: ${pos.get('margin', 0):.2f}"
            )

        lines.append("=" * 55)
        return "\n".join(lines)


class FuturesPaperTrader:
    """Futures paper trading bot."""

    def __init__(self):
        self.signal_gen = SignalGenerator(configs=FUTURES_ALPHA_CONFIGS)
        self.risk_mgr = FuturesRiskManager()
        self.onchain_filter = OnchainSignalFilter()
        self.tg = TelegramAlert()
        self.data_fetcher = BinanceFetcher()
        self._last_data_update = None
        self._last_daily_report = None
        logger.info(
            f"FuturesPaperTrader initialized. "
            f"Leverage: {FUTURES_LEVERAGE}x, Coins: {len(FUTURES_ALPHA_CONFIGS)}"
        )

    def update_data(self):
        now = datetime.now(timezone.utc)
        if self._last_data_update and (now - self._last_data_update).total_seconds() < 6 * 3600:
            return
        try:
            logger.info("Auto-updating OHLCV data...")
            init_db()
            self.data_fetcher.fetch_update()
            self._last_data_update = now
        except Exception as e:
            logger.error(f"Data update failed: {e}")

    def send_daily_report(self):
        now = datetime.now(VN_TIMEZONE)
        if now.hour < 10:
            return
        today = now.strftime("%Y-%m-%d")
        if self._last_daily_report == today:
            return

        current_prices = {}
        for symbol in self.risk_mgr.state["open_positions"]:
            try:
                ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker["last"]
            except Exception:
                pass

        self.tg.send_daily_report(self.risk_mgr.state, current_prices)
        self._last_daily_report = today

    def process_manual_actions(self):
        """Drain user-initiated actions from the dashboard queue."""
        actions = manual_actions.consume("futures")
        for a in actions:
            if a.get("type") != "close":
                continue
            symbol = a["symbol"]
            pct = float(a.get("pct", 1.0))
            if symbol not in self.risk_mgr.state["open_positions"]:
                logger.info(f"Manual close skipped — no open position {symbol}")
                continue
            try:
                ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                price = ticker["last"]
            except Exception as e:
                logger.error(f"Manual close: failed to fetch price {symbol}: {e}")
                continue

            if pct >= 0.999:
                trade = self.risk_mgr.close_position(symbol, price, reason="manual_close")
            else:
                trade = self.risk_mgr._partial_close(symbol, pct, price, "manual_close")
            if trade:
                roe = trade.get("roe", 0)
                self.tg.send(
                    f"👤 <b>🔥[FUT] MANUAL CLOSE {symbol} ({pct*100:.0f}%)</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                )

    def check_and_trade(self):
        now = datetime.now(VN_TIMEZONE).strftime("%Y-%m-%d %H:%M (VN)")
        logger.info(f"{'='*50}")
        logger.info(f"[FUTURES] Signal check at {now}")

        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            self.tg.send(f"⛔ <b>🔥[FUT] Trading blocked</b>\n{reason}")
            return

        # Check stops + take-profits
        current_prices = {}
        for symbol in FUTURES_ALPHA_CONFIGS:
            try:
                ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker["last"]
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")

        stopped = self.risk_mgr.check_stops_and_tp(current_prices)
        for trade in stopped:
            reason = trade.get("reason", "")
            roe = trade.get("roe", 0)
            if "take_profit" in reason:
                self.tg.send(
                    f"🎯 <b>🔥[FUT] TAKE-PROFIT {trade['symbol']}</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                )
            elif reason == "trailing_stop":
                self.tg.send(
                    f"📐 <b>🔥[FUT] TRAILING STOP {trade['symbol']}</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                )
            else:
                self.tg.send(
                    f"🛑 <b>🔥[FUT] STOP-LOSS {trade['symbol']}</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                )

        # Get on-chain regime
        regime = self.onchain_filter.get_current_regime()

        # Generate signals
        signals = self.signal_gen.generate_all()

        for sig in signals:
            symbol = sig["symbol"]
            signal_val = sig["signal"]
            has_position = symbol in self.risk_mgr.state["open_positions"]

            print(f"\n{'-'*50}")
            print(f"  [F] {symbol} @ ${sig['close']:,.4f}")

            # Entry (both long AND short allowed on futures)
            if signal_val != 0 and not has_position:
                can_open, corr_reason = self.risk_mgr.can_trade(symbol=symbol)
                if not can_open:
                    print(f"  [BLOCKED] {corr_reason}")
                    continue

                enhanced = self.onchain_filter.enhance_signal(signal_val, symbol)
                if enhanced["enhanced_signal"] == 0:
                    print(f"  [BLOCKED] Blocked by on-chain filter")
                    continue

                side = "long" if signal_val == 1 else "short"
                pos = self.risk_mgr.open_position(
                    symbol=symbol, side=side,
                    entry_price=sig["close"],
                    stop_loss_pct=sig["stop_loss"],
                    atr_pct=sig.get("atr_pct"),
                    regime=sig.get("regime"),
                    strategy=sig.get("strategy", ""),
                )
                sizing = self.risk_mgr.calculate_position_size(
                    symbol, sig["close"], sig["stop_loss"], side=side
                )

                direction = "LONG" if side == "long" else "SHORT"
                tg_emoji = "\U0001f7e2" if side == "long" else "\U0001f534"

                self.tg.send(
                    f"{tg_emoji} <b>🔥[FUT] {direction} {symbol} ({FUTURES_LEVERAGE}x)</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"Entry: <code>${sig['close']:,.4f}</code>\n"
                    f"Notional: <code>${sizing['size_usdt']:.2f}</code>\n"
                    f"Margin: <code>${sizing['margin']:.2f}</code>\n"
                    f"Stop: <code>${sizing['stop_price']:,.4f}</code>\n"
                    f"Liq: <code>${pos['liq_price']:,.4f}</code>\n"
                    f"Strategy: {sig.get('strategy', '')}"
                )

                tag = "[LONG]" if side == "long" else "[SHORT]"
                print(f"  {tag} OPENED {direction} {FUTURES_LEVERAGE}x: "
                      f"notional=${sizing['size_usdt']:.2f}, margin=${sizing['margin']:.2f}")

            # Exit on signal: gate by explicit exit zone + profit >= 0.8R.
            # Without this gate, every signal=0 tick (most of them) closed positions
            # at tiny profits, never letting TP1/TP2 fire and leaving losers to hit
            # full stop_loss → asymmetric R:R despite high win rate.
            elif signal_val == 0 and has_position:
                pos = self.risk_mgr.state["open_positions"][symbol]
                pnl_pct = (sig["close"] - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "short":
                    pnl_pct = -pnl_pct
                stop_pct = pos.get("stop_loss_pct", 0.025)
                in_exit_zone = "exit zone" in (sig.get("reason") or "").lower()
                profit_threshold = 0.8 * stop_pct  # 0.8R
                should_close = in_exit_zone and pnl_pct >= profit_threshold

                if should_close:
                    trade = self.risk_mgr.close_position(
                        symbol=symbol, exit_price=sig["close"], reason="signal_exit",
                    )
                    if trade:
                        tag = "[WIN]" if trade["pnl_usd"] > 0 else "[LOSS]"
                        roe = trade.get("roe", 0)
                        print(f"  {tag} CLOSED: PnL={trade['pnl_pct']:+.2f}% (ROE={roe:+.1f}%)")
                        tg_emoji = "\u2705" if trade["pnl_usd"] > 0 else "\u274c"
                        self.tg.send(
                            f"{tg_emoji} <b>[FUT] CLOSED {symbol}</b>\n"
                            f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                            f"(ROE: <code>{roe:+.1f}%</code>)\n"
                            f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                        )
                else:
                    print(f"  [HOLD] signal=0 but pnl={pnl_pct*100:+.2f}% < 0.8R "
                          f"({profit_threshold*100:.2f}%) — letting TP/SL manage exit")

            elif has_position:
                pos = self.risk_mgr.state["open_positions"][symbol]
                unrealized = (sig["close"] - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "short":
                    unrealized = -unrealized
                roe = unrealized * FUTURES_LEVERAGE * 100
                print(f"  [HOLD] Holding {pos['side']} {FUTURES_LEVERAGE}x -- "
                      f"Unrealized: {unrealized*100:+.2f}% (ROE: {roe:+.1f}%)")

            else:
                print(f"  [--] No action")

        print(f"\n{self.risk_mgr.get_summary()}")

    def run_continuous(self, interval_hours=0.5):
        print("\n" + "=" * 55)
        print("  FUTURES PAPER TRADING BOT STARTED")
        print(f"  Leverage: {FUTURES_LEVERAGE}x | Margin: {FUTURES_MARGIN_TYPE}")
        print(f"  Checking every {interval_hours} hours")
        print(f"  Coins: {', '.join(FUTURES_ALPHA_CONFIGS.keys())}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 55)

        self.tg.send(
            f"🟢 <b>🔥[FUT] Bot STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Leverage: {FUTURES_LEVERAGE}x\n"
            f"Margin: {FUTURES_MARGIN_TYPE}\n"
            f"Coins: {len(FUTURES_ALPHA_CONFIGS)}\n"
            f"Interval: {interval_hours}h"
        )

        # Start Telegram command listener
        def get_state():
            return self.risk_mgr.state

        def get_prices():
            prices = {}
            for symbol in FUTURES_ALPHA_CONFIGS:
                try:
                    ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker["last"]
                except Exception:
                    pass
            return prices

        # Only start Telegram listener if spot bot is NOT running
        # (both polling getUpdates causes conflict — messages get "eaten")
        spot_state = Path("trading/state.json")
        spot_running = False
        if spot_state.exists():
            try:
                import psutil
                # Check if paper_trader process is running
                for proc in psutil.process_iter(["cmdline"]):
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "paper_trader" in cmdline and "futures" not in cmdline:
                        spot_running = True
                        break
            except ImportError:
                pass

        if spot_running:
            logger.warning(
                "Spot bot detected — Telegram listener DISABLED for futures "
                "(use spot bot for commands to avoid conflict)"
            )
            print("  [!] Telegram commands handled by spot bot (no conflict)")
        else:
            self.tg.start_command_listener(get_state, get_prices)

        error_count = 0
        try:
            while True:
                try:
                    self.update_data()
                    self.send_daily_report()
                    self.check_and_trade()
                    error_count = 0
                    logger.info(f"Sleeping {interval_hours}h until next check...")
                    print(f"\n[...] Next check in {interval_hours} hours...")
                    # Poll manual_actions queue every 30s during the wait window so
                    # dashboard-initiated closes execute within 30s instead of 15min.
                    sleep_until = time.time() + interval_hours * 3600
                    while time.time() < sleep_until:
                        try:
                            self.process_manual_actions()
                        except Exception as ma_err:
                            logger.error(f"Manual action processing error: {ma_err}")
                        remaining = sleep_until - time.time()
                        time.sleep(min(30, max(1, remaining)))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in futures loop ({error_count}): {e}")
                    if error_count >= 5:
                        self.tg.send(
                            f"🚨 <b>🔥[FUT] Bot lỗi liên tục!</b>\n"
                            f"Lỗi: <code>{str(e)[:200]}</code>"
                        )
                        error_count = 0
                    time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n[STOP] Futures bot stopped.")
            self.tg.stop_command_listener()
            self.tg.send("\U0001f534 <b>🔥[FUT] Bot STOPPED</b>")
        except Exception as e:
            self.tg.stop_command_listener()
            self.tg.send("\U0001f480 <b>🔥[FUT] Bot CRASHED!</b>\n<code>{}</code>".format(str(e)[:300]))
            raise

    def show_status(self):
        print(self.risk_mgr.get_summary())

    def show_history(self):
        history = self.risk_mgr.state["trade_history"]
        if not history:
            print("\nNo trades yet.")
            return

        print(f"\n{'=' * 80}")
        print("  FUTURES TRADE HISTORY")
        print(f"{'=' * 80}")
        print(f"  {'#':<4} {'Symbol':<12} {'Side':<6} {'Lev':>4} {'Entry':>10} {'Exit':>10} "
              f"{'PnL %':>8} {'ROE %':>8} {'PnL $':>8}")
        print(f"  {'-'*78}")

        total_pnl = 0
        for i, tr in enumerate(history, 1):
            total_pnl += tr["pnl_usd"]
            tag = "[W]" if tr["pnl_usd"] > 0 else "[L]"
            lev = tr.get("leverage", FUTURES_LEVERAGE)
            roe = tr.get("roe", 0)
            print(
                f"  {tag}{i:<3} {tr['symbol']:<12} {tr['side']:<6} {lev:>3}x "
                f"${tr['entry_price']:>9.4f} ${tr['exit_price']:>9.4f} "
                f"{tr['pnl_pct']:>+7.2f}% {roe:>+7.1f}% ${tr['pnl_usd']:>+7.2f}"
            )

        print(f"  {'-'*78}")
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Capital: ${self.risk_mgr.state['capital']:.2f}")
        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha — Futures Paper Trading")
    parser.add_argument("--once", action="store_true", help="Check signals once")
    parser.add_argument("--run", action="store_true", help="Run continuously")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--reset", action="store_true", help="Reset state")
    parser.add_argument("--interval", type=float, default=0.5, help="Check interval in hours")

    args = parser.parse_args()
    bot = FuturesPaperTrader()

    if args.once:
        bot.check_and_trade()
    elif args.run:
        bot.run_continuous(args.interval)
    elif args.status:
        bot.show_status()
    elif args.history:
        bot.show_history()
    elif args.reset:
        confirm = input("[WARNING] Reset all futures state? (yes/no): ")
        if confirm.lower() == "yes":
            bot.risk_mgr.reset()
            print("[OK] Futures state reset.")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m trading.futures_trader --run              # Run futures bot")
        print("  python -m trading.futures_trader --run --interval 1 # Check every 1h")
        print("  python -m trading.futures_trader --status           # Current state")
        print("  python -m trading.futures_trader --history          # Trade log")


if __name__ == "__main__":
    main()
