"""
Paper Trading Bot
==================
Automated paper trading bot that:
- Checks signals every 4 hours (aligned with candle close)
- Manages positions with stop-loss
- Logs all trades to JSON + console
- Optional Telegram alerts

Usage:
    # Run once (check signals now)
    python -m trading.paper_trader --once

    # Run continuously (check every 4h)
    python -m trading.paper_trader --run

    # Show current status
    python -m trading.paper_trader --status

    # Show trade history
    python -m trading.paper_trader --history

    # Reset all state
    python -m trading.paper_trader --reset
"""

import argparse
import json
import time
import os
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from config.settings import LOG_FILE, UPDATE_INTERVAL_HOURS, VN_TIMEZONE
from trading.signal_generator import SignalGenerator, ALPHA_CONFIGS
from trading.risk_manager import RiskManager
from strategies.onchain_alphas import OnchainSignalFilter
from utils.telegram import TelegramAlert

try:
    from trading.price_monitor import PriceMonitor
except ImportError:
    PriceMonitor = None

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

SIGNAL_LOG_FILE = Path("logs/signal_history.jsonl")


class PaperTrader:
    """Automated paper trading bot."""

    # Cooldown: don't re-enter same symbol+direction within N seconds
    SIGNAL_COOLDOWN_SECS = 4 * 3600  # 4 hours (1 candle)

    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.risk_mgr = RiskManager()
        self.onchain_filter = OnchainSignalFilter()
        self.tg = TelegramAlert()
        self._last_daily_report = None
        self._signal_cooldowns: dict = {}  # {(symbol, direction): timestamp}
        self.price_monitor = None
        logger.info("PaperTrader initialized (with on-chain filter + Telegram).")

    def _is_on_cooldown(self, symbol: str, signal_val: int) -> bool:
        """Check if a signal is on cooldown (duplicate within same candle window)."""
        key = (symbol, signal_val)
        last_ts = self._signal_cooldowns.get(key)
        if last_ts is None:
            return False
        elapsed = time.time() - last_ts
        return elapsed < self.SIGNAL_COOLDOWN_SECS

    def _set_cooldown(self, symbol: str, signal_val: int):
        """Record that a signal was acted upon."""
        self._signal_cooldowns[(symbol, signal_val)] = time.time()
        # Clean up old entries
        now = time.time()
        self._signal_cooldowns = {
            k: v for k, v in self._signal_cooldowns.items()
            if now - v < self.SIGNAL_COOLDOWN_SECS
        }

    def log_signal(self, signal, action, reason="", blocked_reason=""):
        """Write a signal event as a JSON line to SIGNAL_LOG_FILE."""
        SIGNAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": signal.get("symbol", ""),
            "signal": signal.get("signal", 0),
            "strategy": signal.get("strategy", ""),
            "action": action,
            "price": signal.get("close", 0),
            "reason": reason,
            "blocked_reason": blocked_reason,
        }
        with open(SIGNAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def send_daily_report(self):
        """Send a daily report at 10:00 Vietnam time (once per day)."""
        now = datetime.now(VN_TIMEZONE)
        today_key = now.strftime("%Y-%m-%d")
        if now.hour >= 10 and self._last_daily_report != today_key:
            self._last_daily_report = today_key
            current_prices = {}
            for symbol in ALPHA_CONFIGS:
                try:
                    ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                    current_prices[symbol] = ticker["last"]
                except Exception as e:
                    logger.error(f"Daily report: failed to fetch {symbol}: {e}")
            try:
                self.tg.send_daily_report(self.risk_mgr.state, current_prices)
                logger.info("Daily report sent.")
            except Exception as e:
                logger.error(f"Failed to send daily report: {e}")
        
    def check_and_trade(self):
        """Main loop iteration: check signals, manage positions, execute trades."""
        now = datetime.now(VN_TIMEZONE).strftime("%Y-%m-%d %H:%M (VN)")
        logger.info(f"{'='*50}")
        logger.info(f"Signal check at {now}")

        # ── Step 1: General risk checks (symbol-specific check done below) ──
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            print(f"\n[BLOCKED] Trading blocked: {reason}")
            self.tg.send(f"⛔ <b>·[spot] Trading blocked</b>\n{reason}")
            return

        # ── Step 2: Check stop-losses on open positions ──
        current_prices = {}
        for symbol in ALPHA_CONFIGS:
            try:
                ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker["last"]
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")

        stopped = self.risk_mgr.check_stops_and_tp(current_prices)
        for trade in stopped:
            tp_reason = trade.get("reason", "stop_loss")
            if tp_reason == "take_profit_1":
                print(f"\n[TP1] TAKE-PROFIT-1: {trade['symbol']} PnL={trade['pnl_pct']:+.2f}%")
                self.tg.send(
                    f"🎯 <b>Take-Profit-1 hit</b>\n"
                    f"{trade['symbol']} PnL={trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})"
                )
            elif tp_reason == "take_profit_2":
                print(f"\n[TP2] TAKE-PROFIT-2: {trade['symbol']} PnL={trade['pnl_pct']:+.2f}%")
                self.tg.send(
                    f"🎯🎯 <b>Take-Profit-2 hit</b>\n"
                    f"{trade['symbol']} PnL={trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})"
                )
            elif tp_reason == "trailing_stop":
                print(f"\n[TS] TRAILING-STOP: {trade['symbol']} PnL={trade['pnl_pct']:+.2f}%")
                self.tg.send_stop_loss(trade["symbol"], trade["pnl_pct"], trade["pnl_usd"], reason="trailing_stop")
            else:
                print(f"\n[SL] STOP-LOSS: {trade['symbol']} PnL={trade['pnl_pct']:+.2f}%")
                self.tg.send_stop_loss(trade["symbol"], trade["pnl_pct"], trade["pnl_usd"])

        # ── Step 3: Get on-chain regime ──
        regime = self.onchain_filter.get_current_regime()
        print(f"\n  [OnChain] F&G={regime['fng_value']} ({regime['regime']}), "
              f"buy_mult={regime['buy_multiplier']}x, sell_mult={regime['sell_multiplier']}x")

        # ── Step 4: Generate signals ──
        signals = self.signal_gen.generate_all()

        for sig in signals:
            symbol = sig["symbol"]
            signal_val = sig["signal"]
            config = ALPHA_CONFIGS[symbol]
            has_position = symbol in self.risk_mgr.state["open_positions"]

            print(f"\n{'-'*50}")
            print(f"  {symbol} @ ${sig['close']:,.4f}")
            print(f"  ROC(10)={sig['roc_10']:.2f} | RSI={sig['rsi']:.1f} | "
                  f"Vol Ratio={sig['volume_ratio']:.2f}")
            regime = sig.get('regime', 'unknown')
            regime_conf = sig.get('regime_confidence', 0)
            print(f"  Regime: {regime} ({regime_conf:.0%}) | Signal: {sig['reason']}")

            # ── Entry Logic (with on-chain + correlation filter) ──
            if signal_val != 0 and not has_position:
                # Check cooldown (prevent duplicate signals on same candle)
                if self._is_on_cooldown(symbol, signal_val):
                    direction = "LONG" if signal_val == 1 else "SHORT"
                    logger.debug(f"  {symbol}: {direction} signal on cooldown, skipping")
                    print(f"  -- {direction} signal on cooldown, skipping")
                    self.log_signal(sig, "blocked", reason=sig.get("reason", ""), blocked_reason="cooldown")
                    continue

                # Check correlation filter
                can_open, corr_reason = self.risk_mgr.can_trade(symbol=symbol)
                if not can_open:
                    print(f"  [BLOCKED] {corr_reason}")
                    self.log_signal(sig, "blocked", reason=sig.get("reason", ""), blocked_reason=corr_reason)
                    continue

                # Check on-chain filter
                enhanced = self.onchain_filter.enhance_signal(signal_val, symbol)
                print(f"  [Filter] On-chain: {enhanced['reason']}")

                if enhanced["enhanced_signal"] == 0:
                    # Signal blocked by on-chain
                    print(f"  [BLOCKED] Trade blocked by on-chain filter")
                    self.log_signal(sig, "blocked", reason=sig.get("reason", ""), blocked_reason="on-chain filter")
                    continue

                # Spot only supports LONG
                if signal_val != 1:
                    print(f"  [SKIP] Spot does not support SHORT")
                    continue
                side = "long"
                pos = self.risk_mgr.open_position(
                    symbol=symbol,
                    side=side,
                    entry_price=sig["close"],
                    stop_loss_pct=sig["stop_loss"],
                    atr_pct=sig.get("atr_pct"),
                    regime=sig.get("regime"),
                    strategy=sig.get("strategy", ""),
                )
                sizing = self.risk_mgr.calculate_position_size(
                    symbol, sig["close"], sig["stop_loss"]
                )

                emoji = "🟢" if side == "long" else "🔴"
                direction = "LONG" if side == "long" else "SHORT"
                sl_sign = "-" if side == "long" else "+"

                msg = (
                    f"{emoji} <b>{direction} {symbol}</b>\n"
                    f"Entry: ${sig['close']:,.4f}\n"
                    f"Size: ${sizing['size_usdt']:.2f} ({sizing['capital_pct']:.1f}%)\n"
                    f"Stop: ${sizing['stop_price']:,.4f} ({sl_sign}{sig['stop_loss']:.0%})\n"
                    f"On-chain: F&G={enhanced['fng_value']} ({enhanced['regime']}) "
                    f"confidence={enhanced['confidence']:.0%}\n"
                    f"Reason: {sig['reason']}"
                )
                dir_tag = "+" if side == "long" else "-"
                print(f"\n  [{dir_tag}] OPENED {direction}: ${sizing['size_usdt']:.2f} "
                      f"(confidence={enhanced['confidence']:.0%})")
                self.tg.send_trade_opened(
                    symbol=symbol, side=side,
                    entry_price=sig["close"],
                    size_usdt=sizing["size_usdt"],
                    stop_price=sizing["stop_price"],
                    strategy=sig.get("strategy", ""),
                    regime=sig.get("regime", ""),
                    confidence=enhanced.get("confidence", 0),
                )
                self.log_signal(sig, "opened", reason=sig.get("reason", ""))
                self._set_cooldown(symbol, signal_val)

            # ── Exit Logic ──
            elif signal_val == 0 and has_position:
                trade = self.risk_mgr.close_position(
                    symbol=symbol,
                    exit_price=sig["close"],
                    reason="signal_exit",
                )
                if trade:
                    result_tag = "WIN" if trade["pnl_usd"] > 0 else "LOSS"
                    print(f"\n  [{result_tag}] CLOSED: PnL={trade['pnl_pct']:+.2f}%")
                    self.tg.send_trade_closed(
                        symbol=symbol,
                        pnl_pct=trade["pnl_pct"],
                        pnl_usd=trade["pnl_usd"],
                        reason=sig["reason"],
                        capital=self.risk_mgr.state["capital"],
                    )
                    self.log_signal(sig, "closed", reason=sig.get("reason", ""))

            elif has_position:
                pos = self.risk_mgr.state["open_positions"][symbol]
                unrealized = (sig["close"] - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "short":
                    unrealized = -unrealized
                print(f"  [HOLD] {pos['side']} -- Unrealized: {unrealized*100:+.2f}%")
                self.log_signal(sig, "no_action", reason="holding")

            else:
                print(f"  [--] No action")
                self.log_signal(sig, "no_action", reason="no signal")

        # Update WebSocket subscriptions after position changes
        if self.price_monitor:
            self.price_monitor.update_subscriptions()

        # Print summary
        print(f"\n{self.risk_mgr.get_summary()}")

    def run_continuous(self, interval_hours: float = 4):
        """Run bot continuously, checking every N hours."""
        print("\n" + "=" * 50)
        print("  PAPER TRADING BOT STARTED")
        print(f"  Checking every {interval_hours} hours")
        print(f"  Coins: {', '.join(ALPHA_CONFIGS.keys())}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 50)

        self.tg.send_bot_started(list(ALPHA_CONFIGS.keys()))

        # Start real-time price monitor (WebSocket)
        if PriceMonitor is not None:
            try:
                self.price_monitor = PriceMonitor(self.risk_mgr, self.tg)
                self.price_monitor.start()
                logger.info("Real-time price monitor started (WebSocket)")
            except Exception as e:
                logger.warning(f"Price monitor not started: {e}")
                self.price_monitor = None

        # Start Telegram command listener
        try:
            self.tg.start_command_listener(
                state_getter=lambda: self.risk_mgr.state,
                price_getter=lambda: {
                    s: self.signal_gen.exchange.fetch_ticker(s)["last"]
                    for s in ALPHA_CONFIGS
                },
            )
        except Exception as e:
            logger.warning(f"Telegram command listener not started: {e}")

        consecutive_errors = 0
        try:
            while True:
                try:
                    self.check_and_trade()
                    self.send_daily_report()
                    consecutive_errors = 0

                    logger.info(f"Sleeping {interval_hours}h until next check...")
                    print(f"\n[...] Next check in {interval_hours} hours... (Ctrl+C to stop)")
                    time.sleep(interval_hours * 3600)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in main loop ({consecutive_errors}x): {e}")
                    print(f"\n[ERROR] {e}. Retrying in 60s...")
                    if consecutive_errors >= 5:
                        self.tg.send(
                            f"🚨 <b>·[spot] Bot alert</b>\n"
                            f"{consecutive_errors} consecutive errors!\n"
                            f"Last: {e}"
                        )
                    time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n[STOP] Bot stopped by user.")
            if self.price_monitor:
                self.price_monitor.stop()
            self.tg.send_bot_stopped()
        except Exception as e:
            logger.critical(f"Bot crashed: {e}")
            if self.price_monitor:
                self.price_monitor.stop()
            self.tg.send(f"💀 <b>·[spot] Bot CRASHED</b>\n{e}")
            raise

    def show_status(self):
        """Show current trading status."""
        print(self.risk_mgr.get_summary())

    def show_history(self):
        """Show trade history."""
        history = self.risk_mgr.state["trade_history"]
        if not history:
            print("\nNo trades yet.")
            return

        print("\n" + "=" * 70)
        print("  TRADE HISTORY")
        print("=" * 70)
        print(f"  {'#':<4} {'Symbol':<12} {'Side':<6} {'Entry':>10} {'Exit':>10} "
              f"{'PnL %':>8} {'PnL $':>8} {'Reason':<15}")
        print(f"  {'-'*70}")

        total_pnl = 0
        for i, t in enumerate(history, 1):
            total_pnl += t["pnl_usd"]
            tag = "W" if t["pnl_usd"] > 0 else "L"
            print(
                f"  [{tag}]{i:<3} {t['symbol']:<12} {t['side']:<6} "
                f"${t['entry_price']:>9.4f} ${t['exit_price']:>9.4f} "
                f"{t['pnl_pct']:>+7.2f}% ${t['pnl_usd']:>+7.2f} "
                f"{t['reason']:<15}"
            )

        print(f"  {'-'*70}")
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Current Capital: ${self.risk_mgr.state['capital']:.2f}")
        print("=" * 70)


def _show_signal_log(limit=50):
    """Display recent entries from the signal history log."""
    if not SIGNAL_LOG_FILE.exists():
        print("No signal log found.")
        return

    lines = SIGNAL_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(line) for line in lines[-limit:]]

    print("\n" + "=" * 90)
    print("  SIGNAL HISTORY (last {} entries)".format(len(entries)))
    print("=" * 90)
    print(f"  {'Timestamp':<22} {'Symbol':<12} {'Sig':>4} {'Action':<10} {'Price':>12} {'Reason':<20} {'Blocked':<15}")
    print(f"  {'-'*90}")
    for e in entries:
        ts = e.get("timestamp", "")[:19]
        print(
            f"  {ts:<22} {e.get('symbol',''):<12} {e.get('signal',0):>4} "
            f"{e.get('action',''):<10} ${e.get('price',0):>11,.4f} "
            f"{e.get('reason',''):<20} {e.get('blocked_reason',''):<15}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha — Paper Trading Bot")
    parser.add_argument("--once", action="store_true", help="Check signals once and exit")
    parser.add_argument("--run", action="store_true", help="Run continuously (every 4h)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--signal-log", action="store_true", help="Show signal history log")
    parser.add_argument("--daily-report", action="store_true", help="Send daily report now")
    parser.add_argument("--reset", action="store_true", help="Reset all trading state")
    parser.add_argument("--test-telegram", action="store_true", help="Test Telegram connection")
    parser.add_argument("--interval", type=float, default=4, help="Check interval in hours (e.g. 0.5 = 30min)")

    args = parser.parse_args()

    if args.signal_log:
        _show_signal_log()
        return

    bot = PaperTrader()

    if args.test_telegram:
        if bot.tg.test_connection():
            print("[OK] Telegram connected! Check your bot chat.")
        else:
            print("[FAIL] Telegram not configured or connection failed.")
            print("  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    if args.daily_report:
        bot._last_daily_report = None
        bot.send_daily_report()
        print("Daily report sent (if Telegram configured).")
        return

    if args.once:
        bot.check_and_trade()
    elif args.run:
        bot.run_continuous(args.interval)
    elif args.status:
        bot.show_status()
    elif args.history:
        bot.show_history()
    elif args.reset:
        confirm = input("Reset all trading state? (yes/no): ")
        if confirm.lower() == "yes":
            bot.risk_mgr.reset()
            print("[OK] State reset.")
        else:
            print("Cancelled.")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m trading.paper_trader --once        # Check signals now")
        print("  python -m trading.paper_trader --run         # Run 24/7 bot")
        print("  python -m trading.paper_trader --status      # Current state")
        print("  python -m trading.paper_trader --history     # Trade log")
        print("  python -m trading.paper_trader --signal-log  # Signal history")
        print("  python -m trading.paper_trader --daily-report # Force daily report")


if __name__ == "__main__":
    main()
