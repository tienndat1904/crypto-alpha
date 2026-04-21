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
from datetime import datetime, timezone, timedelta

VN_TZ = timezone(timedelta(hours=7))
import threading
import time

import requests
from loguru import logger


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def _position_value(pos: dict) -> float:
    """Equity contribution of an open position: margin for futures, notional for spot."""
    if pos.get("leverage") and pos.get("margin"):
        return pos["margin"]
    return pos.get("size_usdt", 0)


class TelegramAlert:
    """Sends formatted alerts via Telegram bot."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)

        # Command listener state
        self._listener_thread = None
        self._listener_running = False
        self._update_offset = 0
        self._state_getter = None
        self._price_getter = None

        if not self.enabled:
            logger.warning(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable alerts."
            )

    def _call(self, method: str, **kwargs) -> dict:
        """Make a Telegram Bot API call."""
        url = TELEGRAM_API.format(token=self.token, method=method)
        try:
            resp = requests.post(url, json=kwargs, timeout=30)
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Telegram API error: {data.get('description')}")
            return data
        except requests.exceptions.Timeout as e:
            logger.debug(f"Telegram request timed out: {e}")
            return {}
        except Exception as e:
            logger.error(f"Telegram request failed: {e}")
            return {}

    def _call_long_poll(self, method: str, **kwargs) -> dict:
        """Make a Telegram Bot API call with long-poll timeout (for getUpdates)."""
        url = TELEGRAM_API.format(token=self.token, method=method)
        try:
            resp = requests.post(url, json=kwargs, timeout=60)
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Telegram API error: {data.get('description')}")
            return data
        except requests.exceptions.Timeout:
            return {}
        except Exception as e:
            logger.error(f"Telegram long-poll request failed: {e}")
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
            f"{emoji} <b>·[spot] {direction} {symbol}</b>\n"
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

        # Spot alerts silent — futures take notification priority
        self.send(msg, silent=True)

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
            f"{emoji} <b>·[spot] CLOSED {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"PnL: <code>{pnl_pct:+.2f}%</code> (<code>${pnl_usd:+.2f}</code>)\n"
        )
        if reason:
            msg += f"Reason: {reason}\n"
        if capital:
            msg += f"Capital: <code>${capital:.2f}</code>\n"

        self.send(msg, silent=True)

    def send_stop_loss(self, symbol: str, pnl_pct: float, pnl_usd: float, reason: str = "stop_loss"):
        """Send stop-loss or trailing stop alert."""
        if reason == "trailing_stop":
            emoji = "📐"
            label = "TRAILING STOP"
        else:
            emoji = "🛑"
            label = "STOP-LOSS"
        msg = (
            f"{emoji} <b>·[spot] {label} {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"PnL: <code>{pnl_pct:+.2f}%</code> (<code>${pnl_usd:+.2f}</code>)\n"
        )
        self.send(msg)

    def send_breakeven_alert(self, symbol: str, side: str, entry_price: float):
        """Send alert when stop is moved to breakeven."""
        emoji = "🟢" if side == "long" else "🔴"
        msg = (
            f"🔒 <b>Breakeven Stop: {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{emoji} {side.upper()} | Entry: <code>${entry_price:,.4f}</code>\n"
            f"Stop đã dời lên entry (breakeven)\n"
        )
        self.send(msg, silent=True)

    def send_kill_switch(self, drawdown: float, capital: float):
        """Send kill switch alert."""
        msg = (
            f"🚨 <b>·[spot] KILL SWITCH ACTIVATED</b>\n"
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
            total_equity += _position_value(pos)
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

    def send_daily_report(self, state: dict, prices: dict = None):
        """Send full daily portfolio report with HTML formatting."""
        capital = state.get("capital", 0)
        peak = state.get("peak_capital", capital)
        total_trades = state.get("total_trades", 0)
        total_wins = state.get("total_wins", 0)
        total_pnl = state.get("total_pnl", 0)
        open_positions = state.get("open_positions", {})

        # Compute equity
        total_equity = capital
        for pos in open_positions.values():
            total_equity += _position_value(pos)

        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        loss_count = total_trades - total_wins

        timestamp = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S (VN)")

        msg = (
            f"📋 <b>·[spot] BÁO CÁO HÀNG NGÀY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {timestamp}\n\n"
            f"💰 <b>Tài khoản:</b>\n"
            f"  Equity: <code>${total_equity:,.2f}</code>\n"
            f"  Capital: <code>${capital:,.2f}</code>\n"
            f"  Peak: <code>${peak:,.2f}</code>\n"
            f"  Drawdown: <code>{drawdown:.2%}</code>\n\n"
            f"📈 <b>Hiệu suất:</b>\n"
            f"  Tổng giao dịch: {total_trades}\n"
            f"  Thắng/Thua: {total_wins}/{loss_count}\n"
            f"  Win rate: <code>{win_rate:.1f}%</code>\n"
            f"  Tổng PnL: <code>${total_pnl:+,.2f}</code>\n"
        )

        if open_positions:
            msg += f"\n📂 <b>Vị thế đang mở ({len(open_positions)}):</b>\n"
            for sym, pos in open_positions.items():
                side = pos.get("side", "long").upper()
                entry = pos.get("entry_price", 0)
                size = pos.get("size_usdt", 0)

                # Compute unrealized PnL if prices available
                unrealized = 0
                if prices and sym in prices and entry > 0:
                    current = prices[sym]
                    if pos.get("side", "long") == "long":
                        unrealized = (current - entry) / entry * size
                    else:
                        unrealized = (entry - current) / entry * size

                pnl_str = f"${unrealized:+,.2f}" if prices and sym in prices else "N/A"
                msg += f"  {side} {sym} | Entry: ${entry:,.4f} | Size: ${size:.2f} | uPnL: <code>{pnl_str}</code>\n"
        else:
            msg += "\n📂 <b>Không có vị thế đang mở</b>\n"

        self.send(msg)

    def send_bot_started(self, coins: list, strategies: list = None):
        """Send bot started notification with timestamp and details."""
        timestamp = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S (VN)")
        if strategies is None:
            strategies = ["MR", "VB", "MTF"]
        strategies_str = ", ".join(strategies)
        coin_count = len(coins)

        msg = (
            f"🤖 <b>·[spot] Paper Trading Bot Started</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🕐 {timestamp}\n"
            f"Coins ({coin_count}): {', '.join(coins)}\n"
            f"Strategies: {strategies_str}\n"
            f"Features: Regime detection, Correlation filter\n"
        )
        self.send(msg)

    def send_bot_stopped(self):
        """Send bot stopped notification with timestamp."""
        timestamp = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S (VN)")
        msg = (
            f"🛑 <b>·[spot] Paper Trading Bot Stopped</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🕐 {timestamp}\n"
        )
        self.send(msg)

    # ── Interactive Command System ──

    def start_command_listener(self, state_getter, price_getter):
        """
        Start a background thread that polls for Telegram commands.

        Args:
            state_getter: callable that returns the current state dict
            price_getter: callable that returns a dict of {symbol: price}
        """
        self._state_getter = state_getter
        self._price_getter = price_getter

        if not self.enabled:
            logger.warning("Telegram not enabled, command listener not started.")
            return

        self._listener_running = True
        self._listener_thread = threading.Thread(
            target=self._poll_commands, daemon=True, name="tg-cmd-listener"
        )
        self._listener_thread.start()
        logger.info("Telegram command listener started.")

    def stop_command_listener(self):
        """Stop the background command listener thread."""
        self._listener_running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5)
            logger.info("Telegram command listener stopped.")
        self._listener_thread = None

    def _poll_commands(self):
        """Long-polling loop that fetches updates and dispatches commands."""
        while self._listener_running:
            try:
                data = self._call_long_poll(
                    "getUpdates",
                    offset=self._update_offset,
                    timeout=30,
                )
                results = data.get("result", [])
                for update in results:
                    update_id = update.get("update_id", 0)
                    self._update_offset = update_id + 1

                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = str(message.get("chat", {}).get("id", ""))

                    if not text or not chat_id:
                        continue

                    # Only respond to authorized chat_id
                    if chat_id != str(self.chat_id):
                        logger.warning(f"Unauthorized command from chat_id: {chat_id}")
                        continue

                    self._handle_command(text.strip(), chat_id)

            except Exception as e:
                logger.error(f"Command poll error: {e}")
                time.sleep(5)

    def _handle_command(self, text: str, chat_id: str):
        """Handle an incoming command from the authorized user."""
        cmd = text.lower().split()[0] if text else ""

        if cmd in ("/start", "/help"):
            msg = (
                "🤖 <b>Crypto Alpha Bot - Lệnh điều khiển</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "/status - Trạng thái tổng quan\n"
                "/balance - Số dư tài khoản chi tiết\n"
                "/positions - Vị thế đang mở\n"
                "/history - 10 giao dịch gần nhất\n"
                "/pnl - Phân tích lãi/lỗ\n"
                "/report - Báo cáo hàng ngày\n"
                "/help - Hiển thị menu này\n"
            )
            self.send(msg)

        elif cmd == "/status":
            self._cmd_status()

        elif cmd == "/balance":
            self._cmd_balance()

        elif cmd == "/positions":
            self._cmd_positions()

        elif cmd == "/history":
            self._cmd_history()

        elif cmd == "/pnl":
            self._cmd_pnl()

        elif cmd == "/report":
            self._cmd_report()

        else:
            self.send(f"❓ Lệnh không hợp lệ: <code>{cmd}</code>\nGõ /help để xem danh sách lệnh.")

    def _cmd_status(self):
        """Handle /status command."""
        state = self._state_getter() if self._state_getter else {}
        if not state:
            self.send("⚠️ Không có dữ liệu trạng thái.")
            return

        capital = state.get("capital", 0)
        peak = state.get("peak_capital", capital)
        total_trades = state.get("total_trades", 0)
        total_wins = state.get("total_wins", 0)
        total_pnl = state.get("total_pnl", 0)
        open_positions = state.get("open_positions", {})

        total_equity = capital
        for pos in open_positions.values():
            total_equity += _position_value(pos)

        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        pos_count = len(open_positions)

        msg = (
            f"📊 <b>Trạng thái Bot</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Equity: <code>${total_equity:,.2f}</code>\n"
            f"Drawdown: <code>{drawdown:.2%}</code>\n"
            f"Vị thế mở: {pos_count}\n"
            f"Tổng giao dịch: {total_trades}\n"
            f"Win rate: <code>{win_rate:.1f}%</code>\n"
            f"PnL: <code>${total_pnl:+,.2f}</code>\n"
        )
        self.send(msg)

    def _cmd_balance(self):
        """Handle /balance command."""
        state = self._state_getter() if self._state_getter else {}
        if not state:
            self.send("⚠️ Không có dữ liệu trạng thái.")
            return

        capital = state.get("capital", 0)
        peak = state.get("peak_capital", capital)
        initial = state.get("initial_capital", capital)
        total_pnl = state.get("total_pnl", 0)
        open_positions = state.get("open_positions", {})

        total_equity = capital
        in_positions = 0.0
        for pos in open_positions.values():
            value = _position_value(pos)
            total_equity += value
            in_positions += value

        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        roi = (total_equity - initial) / initial * 100 if initial > 0 else 0

        msg = (
            f"💰 <b>Số dư tài khoản</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Vốn ban đầu: <code>${initial:,.2f}</code>\n"
            f"Equity hiện tại: <code>${total_equity:,.2f}</code>\n"
            f"Cash khả dụng: <code>${capital:,.2f}</code>\n"
            f"Đang trong vị thế: <code>${in_positions:,.2f}</code>\n"
            f"Peak equity: <code>${peak:,.2f}</code>\n"
            f"Drawdown: <code>{drawdown:.2%}</code>\n"
            f"ROI: <code>{roi:+.2f}%</code>\n"
            f"Tổng PnL: <code>${total_pnl:+,.2f}</code>\n"
        )
        self.send(msg)

    def _cmd_positions(self):
        """Handle /positions command."""
        state = self._state_getter() if self._state_getter else {}
        prices = self._price_getter() if self._price_getter else {}

        open_positions = state.get("open_positions", {})
        if not open_positions:
            self.send("📂 <b>Không có vị thế đang mở</b>")
            return

        msg = f"📂 <b>Vị thế đang mở ({len(open_positions)})</b>\n━━━━━━━━━━━━━━━━\n"

        for sym, pos in open_positions.items():
            side = pos.get("side", "long")
            side_upper = side.upper()
            entry = pos.get("entry_price", 0)
            size = pos.get("size_usdt", 0)
            stop_price = pos.get("stop_price", 0)
            tp1 = pos.get("tp1_price", 0)
            tp2 = pos.get("tp2_price", 0)
            tp1_hit = pos.get("tp1_hit", False)
            emoji = "🟢" if side == "long" else "🔴"

            # Compute unrealized PnL
            upnl_str = "N/A"
            if prices and sym in prices and entry > 0:
                current = prices[sym]
                if side == "long":
                    unrealized = (current - entry) / entry * size
                else:
                    unrealized = (entry - current) / entry * size
                upnl_str = f"${unrealized:+,.2f}"

            tp_status = "TP1 hit ✓" if tp1_hit else f"TP1: ${tp1:,.4f}"

            msg += (
                f"\n{emoji} <b>{side_upper} {sym}</b>\n"
                f"  Entry: <code>${entry:,.4f}</code>\n"
                f"  Size: <code>${size:.2f}</code>\n"
                f"  SL: <code>${stop_price:,.4f}</code>\n"
                f"  {tp_status} | TP2: <code>${tp2:,.4f}</code>\n"
                f"  uPnL: <code>{upnl_str}</code>\n"
            )

        self.send(msg)

    def _cmd_history(self):
        """Handle /history command - show last 10 trades."""
        state = self._state_getter() if self._state_getter else {}
        trade_log = state.get("trade_history", [])

        if not trade_log:
            self.send("📜 <b>Chưa có giao dịch nào</b>")
            return

        last_10 = trade_log[-10:]
        msg = f"📜 <b>10 giao dịch gần nhất</b>\n━━━━━━━━━━━━━━━━\n"

        for t in reversed(last_10):
            sym = t.get("symbol", "?")
            side = t.get("side", "?").upper()
            pnl = t.get("pnl_usd", 0)
            pnl_pct = t.get("pnl_pct", 0)
            reason = t.get("reason", "")
            emoji = "✅" if pnl >= 0 else "❌"
            reason_str = f" ({reason})" if reason else ""

            msg += f"{emoji} {side} {sym}: <code>${pnl:+,.2f}</code> ({pnl_pct:+.2f}%){reason_str}\n"

        self.send(msg)

    def _cmd_pnl(self):
        """Handle /pnl command - PnL analysis."""
        state = self._state_getter() if self._state_getter else {}
        total_trades = state.get("total_trades", 0)
        total_wins = state.get("total_wins", 0)
        total_pnl = state.get("total_pnl", 0)
        trade_log = state.get("trade_history", [])

        if total_trades == 0:
            self.send("📈 <b>Chưa có dữ liệu PnL</b>")
            return

        loss_count = total_trades - total_wins
        win_rate = total_wins / total_trades * 100

        # Compute average win/loss from trade log
        wins = [t.get("pnl_usd", 0) for t in trade_log if t.get("pnl_usd", 0) > 0]
        losses = [t.get("pnl_usd", 0) for t in trade_log if t.get("pnl_usd", 0) < 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        profit_factor_str = "N/A"
        if losses:
            total_loss_abs = abs(sum(losses))
            if total_loss_abs > 0:
                profit_factor = sum(wins) / total_loss_abs if wins else 0
                profit_factor_str = f"{profit_factor:.2f}"

        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        msg = (
            f"📈 <b>Phân tích PnL</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Tổng giao dịch: {total_trades}\n"
            f"Thắng: {total_wins} | Thua: {loss_count}\n"
            f"Win rate: <code>{win_rate:.1f}%</code>\n"
            f"Tổng PnL: <code>${total_pnl:+,.2f}</code>\n"
            f"PnL TB/lệnh: <code>${avg_pnl:+,.2f}</code>\n\n"
            f"<b>Chi tiết:</b>\n"
            f"  TB thắng: <code>${avg_win:+,.2f}</code>\n"
            f"  TB thua: <code>${avg_loss:+,.2f}</code>\n"
            f"  Lớn nhất thắng: <code>${largest_win:+,.2f}</code>\n"
            f"  Lớn nhất thua: <code>${largest_loss:+,.2f}</code>\n"
            f"  Profit factor: <code>{profit_factor_str}</code>\n"
        )
        self.send(msg)

    def _cmd_report(self):
        """Handle /report command."""
        state = self._state_getter() if self._state_getter else {}
        prices = self._price_getter() if self._price_getter else {}

        if not state:
            self.send("⚠️ Không có dữ liệu để tạo báo cáo.")
            return

        self.send_daily_report(state, prices)
