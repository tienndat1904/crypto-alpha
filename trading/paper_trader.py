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
import time
import os
import sys
sys.path.insert(0, ".")

import requests
from datetime import datetime, timezone
from loguru import logger

from config.settings import LOG_FILE, UPDATE_INTERVAL_HOURS
from trading.signal_generator import SignalGenerator, ALPHA_CONFIGS
from trading.risk_manager import RiskManager
from strategies.onchain_alphas import OnchainSignalFilter

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# ── Telegram Config (optional) ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(message: str):
    """Send message via Telegram bot (if configured)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


class PaperTrader:
    """Automated paper trading bot."""

    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.risk_mgr = RiskManager()
        self.onchain_filter = OnchainSignalFilter()
        logger.info("PaperTrader initialized (with on-chain filter).")
        
    def check_and_trade(self):
        """Main loop iteration: check signals, manage positions, execute trades."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info(f"{'='*50}")
        logger.info(f"Signal check at {now}")

        # ── Step 1: General risk checks (symbol-specific check done below) ──
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            print(f"\n⛔ Trading blocked: {reason}")
            send_telegram(f"⛔ <b>Trading blocked</b>\n{reason}")
            return

        # ── Step 2: Check stop-losses on open positions ──
        current_prices = {}
        for symbol in ALPHA_CONFIGS:
            try:
                ticker = self.signal_gen.exchange.fetch_ticker(symbol)
                current_prices[symbol] = ticker["last"]
            except Exception as e:
                logger.error(f"Failed to fetch price for {symbol}: {e}")

        stopped = self.risk_mgr.check_stop_losses(current_prices)
        for trade in stopped:
            msg = (
                f"🛑 <b>STOP-LOSS HIT</b>\n"
                f"Symbol: {trade['symbol']}\n"
                f"PnL: {trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})"
            )
            print(f"\n🛑 STOP-LOSS: {trade['symbol']} PnL={trade['pnl_pct']:+.2f}%")
            send_telegram(msg)

        # ── Step 3: Get on-chain regime ──
        regime = self.onchain_filter.get_current_regime()
        print(f"\n  📡 On-chain: F&G={regime['fng_value']} ({regime['regime']}), "
              f"buy_mult={regime['buy_multiplier']}x, sell_mult={regime['sell_multiplier']}x")

        # ── Step 4: Generate signals ──
        signals = self.signal_gen.generate_all()

        for sig in signals:
            symbol = sig["symbol"]
            signal_val = sig["signal"]
            config = ALPHA_CONFIGS[symbol]
            has_position = symbol in self.risk_mgr.state["open_positions"]

            print(f"\n{'─'*50}")
            print(f"  {symbol} @ ${sig['close']:,.4f}")
            print(f"  ROC(10)={sig['roc_10']:.2f} | RSI={sig['rsi']:.1f} | "
                  f"Vol Ratio={sig['volume_ratio']:.2f}")
            regime = sig.get('regime', 'unknown')
            regime_conf = sig.get('regime_confidence', 0)
            print(f"  Regime: {regime} ({regime_conf:.0%}) | Signal: {sig['reason']}")

            # ── Entry Logic (with on-chain + correlation filter) ──
            if signal_val != 0 and not has_position:
                # Check correlation filter
                can_open, corr_reason = self.risk_mgr.can_trade(symbol=symbol)
                if not can_open:
                    print(f"  ⛔ {corr_reason}")
                    continue

                # Check on-chain filter
                enhanced = self.onchain_filter.enhance_signal(signal_val, symbol)
                print(f"  🔍 On-chain filter: {enhanced['reason']}")

                if enhanced["enhanced_signal"] == 0:
                    # Signal blocked by on-chain
                    print(f"  ⛔ Trade BLOCKED by on-chain filter")
                    continue

                side = "long" if signal_val == 1 else "short"
                pos = self.risk_mgr.open_position(
                    symbol=symbol,
                    side=side,
                    entry_price=sig["close"],
                    stop_loss_pct=sig["stop_loss"],
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
                print(f"\n  {emoji} OPENED {direction}: ${sizing['size_usdt']:.2f} "
                      f"(confidence={enhanced['confidence']:.0%})")
                send_telegram(msg)

            # ── Exit Logic ──
            elif signal_val == 0 and has_position:
                trade = self.risk_mgr.close_position(
                    symbol=symbol,
                    exit_price=sig["close"],
                    reason="signal_exit",
                )
                if trade:
                    emoji = "✅" if trade["pnl_usd"] > 0 else "❌"
                    msg = (
                        f"{emoji} <b>CLOSED {symbol}</b>\n"
                        f"PnL: {trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})\n"
                        f"Reason: {sig['reason']}"
                    )
                    print(f"\n  {emoji} CLOSED: PnL={trade['pnl_pct']:+.2f}%")
                    send_telegram(msg)

            elif has_position:
                pos = self.risk_mgr.state["open_positions"][symbol]
                unrealized = (sig["close"] - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "short":
                    unrealized = -unrealized
                print(f"  📊 Holding {pos['side']} — Unrealized: {unrealized*100:+.2f}%")

            else:
                print(f"  ⚪ No action")

        # Print summary
        print(f"\n{self.risk_mgr.get_summary()}")

    def run_continuous(self, interval_hours: int = 4):
        """Run bot continuously, checking every N hours."""
        print("\n" + "=" * 50)
        print("  PAPER TRADING BOT STARTED")
        print(f"  Checking every {interval_hours} hours")
        print(f"  Coins: {', '.join(ALPHA_CONFIGS.keys())}")
        print(f"  Press Ctrl+C to stop")
        print("=" * 50)

        send_telegram(
            "🤖 <b>Paper Trading Bot Started</b>\n"
            f"Coins: {', '.join(ALPHA_CONFIGS.keys())}\n"
            f"Interval: {interval_hours}h"
        )

        while True:
            try:
                self.check_and_trade()
                logger.info(f"Sleeping {interval_hours}h until next check...")
                print(f"\n⏳ Next check in {interval_hours} hours... (Ctrl+C to stop)")
                time.sleep(interval_hours * 3600)
            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user.")
                send_telegram("🛑 Paper Trading Bot stopped.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\n⚠️ Error: {e}. Retrying in 60s...")
                time.sleep(60)

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
        print(f"  {'─'*70}")

        total_pnl = 0
        for i, t in enumerate(history, 1):
            total_pnl += t["pnl_usd"]
            emoji = "✅" if t["pnl_usd"] > 0 else "❌"
            print(
                f"  {emoji}{i:<3} {t['symbol']:<12} {t['side']:<6} "
                f"${t['entry_price']:>9.4f} ${t['exit_price']:>9.4f} "
                f"{t['pnl_pct']:>+7.2f}% ${t['pnl_usd']:>+7.2f} "
                f"{t['reason']:<15}"
            )

        print(f"  {'─'*70}")
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Current Capital: ${self.risk_mgr.state['capital']:.2f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha — Paper Trading Bot")
    parser.add_argument("--once", action="store_true", help="Check signals once and exit")
    parser.add_argument("--run", action="store_true", help="Run continuously (every 4h)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--reset", action="store_true", help="Reset all trading state")
    parser.add_argument("--interval", type=int, default=4, help="Check interval in hours")

    args = parser.parse_args()
    bot = PaperTrader()

    if args.once:
        bot.check_and_trade()
    elif args.run:
        bot.run_continuous(args.interval)
    elif args.status:
        bot.show_status()
    elif args.history:
        bot.show_history()
    elif args.reset:
        confirm = input("⚠️  Reset all trading state? (yes/no): ")
        if confirm.lower() == "yes":
            bot.risk_mgr.reset()
            print("✓ State reset.")
        else:
            print("Cancelled.")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m trading.paper_trader --once      # Check signals now")
        print("  python -m trading.paper_trader --run        # Run 24/7 bot")
        print("  python -m trading.paper_trader --status     # Current state")
        print("  python -m trading.paper_trader --history    # Trade log")


if __name__ == "__main__":
    main()
