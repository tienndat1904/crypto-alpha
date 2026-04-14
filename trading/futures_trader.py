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
)
from trading.signal_generator import SignalGenerator, ALPHA_CONFIGS
from trading.risk_manager import RiskManager
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

    def calculate_position_size(self, symbol, entry_price, stop_loss_pct):
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

    def open_position(self, symbol, side, entry_price, stop_loss_pct):
        """Open a futures position (long or short)."""
        sizing = self.calculate_position_size(symbol, entry_price, stop_loss_pct)

        risk_distance = entry_price * stop_loss_pct
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
            "stop_loss_pct": stop_loss_pct,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "tp1_hit": False,
            "liq_price": liq_price,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "funding_paid": 0.0,
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

        trade_record = {
            **pos,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct * 100, 3),
            "pnl_usd": round(pnl_usd, 2),
            "roe": round(roe, 2),  # Return on Equity (leveraged)
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["trade_history"].append(trade_record)
        del self.state["open_positions"][symbol]
        self._save_state()

        emoji = "✅" if pnl_usd > 0 else "❌"
        logger.info(
            f"{emoji} CLOSED {symbol} (Futures): {reason}, "
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
        self.signal_gen = SignalGenerator()
        self.risk_mgr = FuturesRiskManager()
        self.onchain_filter = OnchainSignalFilter()
        self.tg = TelegramAlert()
        self.data_fetcher = BinanceFetcher()
        self._last_data_update = None
        self._last_daily_report = None
        logger.info(
            f"FuturesPaperTrader initialized. "
            f"Leverage: {FUTURES_LEVERAGE}x, Coins: {len(ALPHA_CONFIGS)}"
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
        now = datetime.now(timezone.utc)
        if now.hour < 8:
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

    def check_and_trade(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info(f"{'='*50}")
        logger.info(f"[FUTURES] Signal check at {now}")

        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            self.tg.send(f"⛔ <b>[Futures] Trading blocked</b>\n{reason}")
            return

        # Check stops + take-profits
        current_prices = {}
        for symbol in ALPHA_CONFIGS:
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
                    f"🎯 <b>[Futures] TAKE-PROFIT {trade['symbol']}</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                )
            else:
                self.tg.send(
                    f"🛑 <b>[Futures] STOP {trade['symbol']}</b>\n"
                    f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                    f"(ROE: <code>{roe:+.1f}%</code>)\n"
                    f"(<code>${trade['pnl_usd']:+.2f}</code>)\n"
                    f"Reason: {reason}"
                )

        # Get on-chain regime
        regime = self.onchain_filter.get_current_regime()

        # Generate signals
        signals = self.signal_gen.generate_all()

        for sig in signals:
            symbol = sig["symbol"]
            signal_val = sig["signal"]
            has_position = symbol in self.risk_mgr.state["open_positions"]

            print(f"\n{'─'*50}")
            print(f"  [F] {symbol} @ ${sig['close']:,.4f}")

            # Entry (both long AND short allowed on futures)
            if signal_val != 0 and not has_position:
                can_open, corr_reason = self.risk_mgr.can_trade(symbol=symbol)
                if not can_open:
                    print(f"  ⛔ {corr_reason}")
                    continue

                enhanced = self.onchain_filter.enhance_signal(signal_val, symbol)
                if enhanced["enhanced_signal"] == 0:
                    print(f"  ⛔ Blocked by on-chain filter")
                    continue

                side = "long" if signal_val == 1 else "short"
                pos = self.risk_mgr.open_position(
                    symbol=symbol, side=side,
                    entry_price=sig["close"],
                    stop_loss_pct=sig["stop_loss"],
                )
                sizing = self.risk_mgr.calculate_position_size(
                    symbol, sig["close"], sig["stop_loss"]
                )

                emoji = "🟢" if side == "long" else "🔴"
                direction = "LONG" if side == "long" else "SHORT"

                self.tg.send(
                    f"{emoji} <b>[Futures] {direction} {symbol} ({FUTURES_LEVERAGE}x)</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"Entry: <code>${sig['close']:,.4f}</code>\n"
                    f"Notional: <code>${sizing['size_usdt']:.2f}</code>\n"
                    f"Margin: <code>${sizing['margin']:.2f}</code>\n"
                    f"Stop: <code>${sizing['stop_price']:,.4f}</code>\n"
                    f"Liq: <code>${pos['liq_price']:,.4f}</code>\n"
                    f"Strategy: {sig.get('strategy', '')}"
                )

                print(f"  {emoji} OPENED {direction} {FUTURES_LEVERAGE}x: "
                      f"notional=${sizing['size_usdt']:.2f}, margin=${sizing['margin']:.2f}")

            # Exit
            elif signal_val == 0 and has_position:
                trade = self.risk_mgr.close_position(
                    symbol=symbol, exit_price=sig["close"], reason="signal_exit",
                )
                if trade:
                    emoji = "✅" if trade["pnl_usd"] > 0 else "❌"
                    roe = trade.get("roe", 0)
                    print(f"  {emoji} CLOSED: PnL={trade['pnl_pct']:+.2f}% (ROE={roe:+.1f}%)")
                    self.tg.send(
                        f"{emoji} <b>[Futures] CLOSED {symbol}</b>\n"
                        f"PnL: <code>{trade['pnl_pct']:+.2f}%</code> "
                        f"(ROE: <code>{roe:+.1f}%</code>)\n"
                        f"(<code>${trade['pnl_usd']:+.2f}</code>)"
                    )

            elif has_position:
                pos = self.risk_mgr.state["open_positions"][symbol]
                unrealized = (sig["close"] - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "short":
                    unrealized = -unrealized
                roe = unrealized * FUTURES_LEVERAGE * 100
                print(f"  📊 Holding {pos['side']} {FUTURES_LEVERAGE}x — "
                      f"Unrealized: {unrealized*100:+.2f}% (ROE: {roe:+.1f}%)")

            else:
                print(f"  ⚪ No action")

        print(f"\n{self.risk_mgr.get_summary()}")

    def run_continuous(self, interval_hours=0.5):
        print("\n" + "=" * 55)
        print("  FUTURES PAPER TRADING BOT STARTED")
        print(f"  Leverage: {FUTURES_LEVERAGE}x | Margin: {FUTURES_MARGIN_TYPE}")
        print(f"  Checking every {interval_hours} hours")
        print(f"  Coins: {', '.join(ALPHA_CONFIGS.keys())}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 55)

        self.tg.send(
            f"🟢 <b>[Futures] Bot STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Leverage: {FUTURES_LEVERAGE}x\n"
            f"Margin: {FUTURES_MARGIN_TYPE}\n"
            f"Coins: {len(ALPHA_CONFIGS)}\n"
            f"Interval: {interval_hours}h"
        )

        # Start Telegram command listener
        def get_state():
            return self.risk_mgr.state

        def get_prices():
            prices = {}
            for symbol in ALPHA_CONFIGS:
                try:
                    ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker["last"]
                except Exception:
                    pass
            return prices

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
                    print(f"\n⏳ Next check in {interval_hours} hours...")
                    time.sleep(interval_hours * 3600)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in futures loop ({error_count}): {e}")
                    if error_count >= 5:
                        self.tg.send(
                            f"🚨 <b>[Futures] Bot lỗi liên tục!</b>\n"
                            f"Lỗi: <code>{str(e)[:200]}</code>"
                        )
                        error_count = 0
                    time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n🛑 Futures bot stopped.")
            self.tg.stop_command_listener()
            self.tg.send("🔴 <b>[Futures] Bot STOPPED</b>")
        except Exception as e:
            self.tg.stop_command_listener()
            self.tg.send(f"💀 <b>[Futures] Bot CRASHED!</b>\n<code>{str(e)[:300]}</code>")
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
        print(f"  {'─'*78}")

        total_pnl = 0
        for i, tr in enumerate(history, 1):
            total_pnl += tr["pnl_usd"]
            emoji = "✅" if tr["pnl_usd"] > 0 else "❌"
            lev = tr.get("leverage", FUTURES_LEVERAGE)
            roe = tr.get("roe", 0)
            print(
                f"  {emoji}{i:<3} {tr['symbol']:<12} {tr['side']:<6} {lev:>3}x "
                f"${tr['entry_price']:>9.4f} ${tr['exit_price']:>9.4f} "
                f"{tr['pnl_pct']:>+7.2f}% {roe:>+7.1f}% ${tr['pnl_usd']:>+7.2f}"
            )

        print(f"  {'─'*78}")
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
        confirm = input("⚠️  Reset all futures state? (yes/no): ")
        if confirm.lower() == "yes":
            bot.risk_mgr.reset()
            print("✓ Futures state reset.")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m trading.futures_trader --run              # Run futures bot")
        print("  python -m trading.futures_trader --run --interval 1 # Check every 1h")
        print("  python -m trading.futures_trader --status           # Current state")
        print("  python -m trading.futures_trader --history          # Trade log")


if __name__ == "__main__":
    main()
