"""
Bot Watchdog — Auto-restart & Crash Alert
==========================================
Monitors paper_trader and futures_trader processes.
Restarts them if they crash, sends Telegram alert.

Usage:
    python bot_watchdog.py

Add to Windows Task Scheduler to run at startup:
    Action: Start a program
    Program: python
    Arguments: D:\crypto-alpha\bot_watchdog.py
    Start in: D:\crypto-alpha
"""

import subprocess
import time
import sys
import os

sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
from utils.telegram import TelegramAlert

VN_TZ = timezone(timedelta(hours=7))

# Bot configurations
BOTS = [
    {
        "name": "Spot Paper Trader",
        "tag": "paper_trader",
        "cmd": [sys.executable, "-m", "trading.paper_trader", "--run", "--interval", "0.25"],
        "process": None,
    },
    {
        "name": "Futures Trader",
        "tag": "futures_trader",
        "cmd": [sys.executable, "-m", "trading.futures_trader", "--run", "--interval", "0.25"],
        "process": None,
    },
]

CHECK_INTERVAL = 30  # Check every 30 seconds
MAX_RESTARTS_PER_HOUR = 5


class Watchdog:
    def __init__(self):
        self.tg = TelegramAlert()
        self.restart_counts = {bot["tag"]: [] for bot in BOTS}

    def _now_str(self):
        return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S (VN)")

    def _clean_old_restarts(self, tag):
        """Remove restart records older than 1 hour."""
        cutoff = time.time() - 3600
        self.restart_counts[tag] = [t for t in self.restart_counts[tag] if t > cutoff]

    def start_bot(self, bot, reason="initial_start"):
        """Start a bot process."""
        tag = bot["tag"]
        name = bot["name"]

        # Check restart limit
        self._clean_old_restarts(tag)
        if len(self.restart_counts[tag]) >= MAX_RESTARTS_PER_HOUR:
            msg = (
                f"🚨 <b>[Watchdog] {name} quá giới hạn restart!</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🕐 {self._now_str()}\n"
                f"Đã restart {MAX_RESTARTS_PER_HOUR} lần trong 1 giờ.\n"
                f"Bot bị TẠM DỪNG. Kiểm tra lỗi thủ công."
            )
            self.tg.send(msg)
            print(f"[WATCHDOG] {name}: too many restarts, pausing")
            return False

        try:
            proc = subprocess.Popen(
                bot["cmd"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            bot["process"] = proc
            self.restart_counts[tag].append(time.time())

            if reason == "crash_restart":
                msg = (
                    f"🔄 <b>[Watchdog] {name} đã RESTART</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"🕐 {self._now_str()}\n"
                    f"PID mới: {proc.pid}\n"
                    f"Restart lần {len(self.restart_counts[tag])} trong giờ qua"
                )
                self.tg.send(msg)

            print(f"[WATCHDOG] {name} started (PID {proc.pid}, reason={reason})")
            return True
        except Exception as e:
            msg = (
                f"💀 <b>[Watchdog] Không thể khởi động {name}!</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🕐 {self._now_str()}\n"
                f"Lỗi: <code>{str(e)[:200]}</code>"
            )
            self.tg.send(msg)
            print(f"[WATCHDOG] Failed to start {name}: {e}")
            return False

    def check_bot(self, bot):
        """Check if bot is still running, restart if crashed."""
        proc = bot["process"]
        if proc is None:
            return

        retcode = proc.poll()
        if retcode is not None:
            # Process has exited
            name = bot["name"]
            tag = bot["tag"]

            if retcode == 0:
                reason_text = "thoát bình thường (code 0)"
            else:
                reason_text = f"CRASH (exit code {retcode})"

            msg = (
                f"💀 <b>[Watchdog] {name} đã TẮT!</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🕐 {self._now_str()}\n"
                f"Nguyên nhân: {reason_text}\n"
                f"Đang khởi động lại..."
            )
            self.tg.send(msg)
            print(f"[WATCHDOG] {name} exited (code {retcode}), restarting...")

            bot["process"] = None
            self.start_bot(bot, reason="crash_restart")

    def run(self):
        """Main watchdog loop."""
        print("=" * 50)
        print("  WATCHDOG STARTED")
        print(f"  Monitoring {len(BOTS)} bots")
        print(f"  Check interval: {CHECK_INTERVAL}s")
        print(f"  Max restarts/hour: {MAX_RESTARTS_PER_HOUR}")
        print("=" * 50)

        self.tg.send(
            f"🐕 <b>[Watchdog] STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🕐 {self._now_str()}\n"
            f"Giám sát: {', '.join(b['name'] for b in BOTS)}\n"
            f"Check: mỗi {CHECK_INTERVAL}s\n"
            f"Max restart: {MAX_RESTARTS_PER_HOUR}/giờ"
        )

        # Start all bots
        for bot in BOTS:
            self.start_bot(bot, reason="initial_start")
            time.sleep(2)  # Stagger starts

        # Monitor loop
        try:
            while True:
                for bot in BOTS:
                    self.check_bot(bot)
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\n[WATCHDOG] Stopping...")
            self.tg.send(
                f"🛑 <b>[Watchdog] STOPPED</b>\n"
                f"🕐 {self._now_str()}\n"
                f"Tất cả bot sẽ bị tắt."
            )
            # Kill child processes
            for bot in BOTS:
                if bot["process"] and bot["process"].poll() is None:
                    bot["process"].terminate()
                    print(f"  Terminated {bot['name']}")
            print("[WATCHDOG] Done.")


if __name__ == "__main__":
    watchdog = Watchdog()
    watchdog.run()
