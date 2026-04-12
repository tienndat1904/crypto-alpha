"""
Telegram Alert System
======================
Centralized Telegram bot for trading alerts.

Setup:
  1. Create a bot via @BotFather on Telegram
  2. Get the bot token
  3. Start a chat with your bot, send /start
  4. Get your chat_id via: https://api.telegram.org/bot<TOKEN>/getUpdates
  5. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

Usage:
    from utils.telegram import TelegramAlert
    tg = TelegramAlert()
    tg.send("Hello from Crypto Alpha!")
    tg.send_trade_opened("BTC/USDT", "long", 50000, 200, 48500)
"""

import os
import requests
from loguru import logger


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlert:
    """Sends formatted alerts via Telegram bot."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)

        if not self.enabled:
            logger.warning(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable alerts."
            )

    def _call(self, method: str, **kwargs) -> dict:
        """Make a Telegram Bot API call."""
        url = TELEGRAM_API.format(token=self.token, method=method)
        try:
            resp = requests.post(url, json=kwargs, timeout=10)
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Telegram API error: {data.get('description')}")
            return data
        except Exception as e:
            logger.error(f"Telegram request failed: {e}")
            return {}

    def send(self, message: str, silent: bool = False) -> bool:
        """
        Send a text message.

        Args:
            message: HTML-formatted message text
            silent: If True, send without notification sound
        """
        if not self.enabled:
            return False

        result = self._call(
            "sendMessage",
            chat_id=self.chat_id,
            text=message,
            parse_mode="HTML",
            disable_notification=silent,
        )
        return result.get("ok", False)

    def test_connection(self) -> bool:
        """Test if bot token and chat_id are valid."""
        if not self.enabled:
            return False

        result = self._call("getMe")
        if result.get("ok"):
            bot_name = result["result"]["username"]
            self.send(f"🤖 <b>Crypto Alpha Bot Connected</b>\nBot: @{bot_name}")
            logger.info(f"Telegram connected: @{bot_name}")
            return True
        return False

    # ── Formatted Alerts ──

    def send_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size_usdt: float,
        stop_price: float,
        strategy: str = "",
        regime: str = "",
        confidence: float = 0,
    ):
        """Send trade opened alert."""
        emoji = "🟢" if side == "long" else "🔴"
        direction = "LONG" if side == "long" else "SHORT"

        msg = (
            f"{emoji} <b>{direction} {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Entry: <code>${entry_price:,.4f}</code>\n"
            f"Size: <code>${size_usdt:.2f}</code>\n"
            f"Stop: <code>${stop_price:,.4f}</code>\n"
        )
        if strategy:
            msg += f"Strategy: {strategy}\n"
        if regime:
            msg += f"Regime: {regime}"
            if confidence:
                msg += f" ({confidence:.0%})"
            msg += "\n"

        self.send(msg)

    def send_trade_closed(
        self,
        symbol: str,
        pnl_pct: float,
        pnl_usd: float,
        reason: str = "",
        capital: float = 0,
    ):
        """Send trade closed alert."""
        emoji = "✅" if pnl_usd > 0 else "❌"

        msg = (
            f"{emoji} <b>CLOSED {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"PnL: <code>{pnl_pct:+.2f}%</code> (<code>${pnl_usd:+.2f}</code>)\n"
        )
        if reason:
            msg += f"Reason: {reason}\n"
        if capital:
            msg += f"Capital: <code>${capital:.2f}</code>\n"

        self.send(msg)

    def send_stop_loss(self, symbol: str, pnl_pct: float, pnl_usd: float):
        """Send stop-loss alert."""
        msg = (
            f"🛑 <b>STOP-LOSS {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"PnL: <code>{pnl_pct:+.2f}%</code> (<code>${pnl_usd:+.2f}</code>)\n"
        )
        self.send(msg)

    def send_kill_switch(self, drawdown: float, capital: float):
        """Send kill switch alert."""
        msg = (
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Drawdown: <code>{drawdown:.1%}</code>\n"
            f"Capital: <code>${capital:.2f}</code>\n"
            f"\n⛔ Trading halted. Review system."
        )
        self.send(msg)

    def send_regime_change(self, symbol: str, old_regime: str, new_regime: str):
        """Send regime change alert."""
        emojis = {"trending": "📈", "sideways": "↔️", "choppy": "🌊"}
        emoji = emojis.get(new_regime, "❓")

        msg = (
            f"{emoji} <b>Regime Change: {symbol}</b>\n"
            f"{old_regime.upper()} → {new_regime.upper()}\n"
        )
        self.send(msg, silent=True)

    def send_status(self, state: dict):
        """Send current status summary."""
        capital = state["capital"]
        peak = state["peak_capital"]
        total_equity = capital
        for pos in state["open_positions"].values():
            total_equity += pos["size_usdt"]
        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        win_rate = (
            state["total_wins"] / state["total_trades"] * 100
            if state["total_trades"] > 0 else 0
        )

        msg = (
            f"📊 <b>Status Update</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Equity: <code>${total_equity:.2f}</code>\n"
            f"Drawdown: <code>{drawdown:.1%}</code>\n"
            f"Open: {len(state['open_positions'])}/3\n"
            f"Trades: {state['total_trades']} "
            f"(Win: {win_rate:.0f}%)\n"
            f"PnL: <code>${state['total_pnl']:+.2f}</code>\n"
        )

        if state["open_positions"]:
            msg += "\n<b>Open Positions:</b>\n"
            for sym, pos in state["open_positions"].items():
                msg += f"  {pos['side'].upper()} {sym} @ ${pos['entry_price']:,.4f}\n"

        self.send(msg)

    def send_bot_started(self, coins: list):
        """Send bot started notification."""
        msg = (
            f"🤖 <b>Paper Trading Bot Started</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Coins: {', '.join(coins)}\n"
            f"Strategies: MR, VB, MTF\n"
            f"Features: Regime detection, Correlation filter\n"
        )
        self.send(msg)

    def send_bot_stopped(self):
        """Send bot stopped notification."""
        self.send("🛑 <b>Paper Trading Bot Stopped</b>")
