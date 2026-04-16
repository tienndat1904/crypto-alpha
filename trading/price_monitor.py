"""
Real-Time Price Monitor
========================
WebSocket-based price monitor using Binance streams.
Checks trailing stops and take-profit levels continuously
instead of waiting for the 15-minute polling cycle.

Usage:
    from trading.price_monitor import PriceMonitor
    monitor = PriceMonitor(risk_manager, telegram_bot)
    monitor.start()
    # ... when positions change ...
    monitor.update_subscriptions()
    # ... on shutdown ...
    monitor.stop()
"""

import asyncio
import json
import threading
import time
from loguru import logger

try:
    import websockets
except ImportError:
    websockets = None
    logger.warning(
        "websockets not installed. Run: pip install websockets"
    )


BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"

# Reconnection settings
RECONNECT_DELAY_INITIAL = 1.0   # seconds
RECONNECT_DELAY_MAX = 60.0      # seconds
RECONNECT_DELAY_FACTOR = 2.0    # exponential backoff multiplier


def _symbol_to_stream(symbol: str) -> str:
    """Convert 'BTC/USDT' to 'btcusdt@miniTicker'."""
    return symbol.replace("/", "").lower() + "@miniTicker"


def _stream_to_symbol(stream: str) -> str:
    """
    Convert 'btcusdt@miniTicker' to a matching key from open_positions.
    Returns the raw lowercase pair (e.g. 'btcusdt') so the caller
    can match against the position dict.
    """
    return stream.split("@")[0]


class PriceMonitor:
    """
    Monitors real-time prices via Binance WebSocket and triggers
    stop-loss / take-profit checks immediately.
    """

    def __init__(self, risk_manager, telegram_bot):
        """
        Args:
            risk_manager: RiskManager instance with open_positions and
                          check_stops_and_tp / close_position methods.
            telegram_bot: TelegramAlert instance for sending alerts.
        """
        if websockets is None:
            raise ImportError(
                "websockets library is required. Install with: pip install websockets"
            )

        self.risk_manager = risk_manager
        self.telegram = telegram_bot

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._ws = None

        # Track which symbols we are currently subscribed to
        self._subscribed_symbols: set[str] = set()

        # Lock to serialise subscription updates and stop-check logic
        self._lock = threading.Lock()

        # Map lowercase pair -> original symbol key used in open_positions
        # e.g. "btcusdt" -> "BTC/USDT"
        self._pair_to_symbol: dict[str, str] = {}

        # Latest prices received (can be read externally)
        self.latest_prices: dict[str, float] = {}

    # ── Public API ──────────────────────────────────────────────

    def start(self):
        """Start the price monitor in a background daemon thread."""
        if self._running:
            logger.warning("PriceMonitor is already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="price-monitor",
        )
        self._thread.start()
        logger.info("PriceMonitor started (background thread).")

    def stop(self):
        """Stop the price monitor gracefully."""
        if not self._running:
            return

        self._running = False

        # Cancel all tasks on the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        self._thread = None
        self._loop = None
        logger.info("PriceMonitor stopped.")

    def update_subscriptions(self):
        """
        Called externally when positions change (opened / closed).
        Schedules a subscription update on the event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._sync_subscriptions(), self._loop
            )

    # ── Internal: Thread & Event Loop ───────────────────────────

    def _run_loop(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            if self._running:
                logger.error(f"PriceMonitor loop crashed: {e}")
        finally:
            self._loop.close()

    async def _connect_loop(self):
        """
        Maintain a persistent WebSocket connection with automatic
        reconnection using exponential backoff.
        """
        delay = RECONNECT_DELAY_INITIAL

        while self._running:
            try:
                await self._run_websocket()
                # If _run_websocket returns cleanly (e.g. we stopped),
                # reset the delay.
                delay = RECONNECT_DELAY_INITIAL
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    f"WebSocket disconnected: {e}. "
                    f"Reconnecting in {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * RECONNECT_DELAY_FACTOR, RECONNECT_DELAY_MAX)

    async def _run_websocket(self):
        """
        Open a combined-stream WebSocket, subscribe to relevant
        symbols, and process incoming messages.
        """
        logger.info(f"Connecting to Binance WebSocket: {BINANCE_WS_URL}")

        async with websockets.connect(
            BINANCE_WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            logger.info("Binance WebSocket connected.")

            # Subscribe to symbols that currently have open positions
            await self._sync_subscriptions()

            # Read messages until disconnection or stop
            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    self._handle_message(raw_msg)
                except Exception as e:
                    logger.error(f"Error handling WS message: {e}")

        self._ws = None

    # ── Internal: Subscription Management ───────────────────────

    def _get_desired_symbols(self) -> set[str]:
        """Return the set of symbols that currently have open positions."""
        return set(self.risk_manager.state.get("open_positions", {}).keys())

    async def _sync_subscriptions(self):
        """
        Compare currently subscribed symbols with open positions and
        send SUBSCRIBE / UNSUBSCRIBE requests as needed.
        """
        desired = self._get_desired_symbols()

        # Rebuild pair -> symbol map
        self._pair_to_symbol = {
            sym.replace("/", "").lower(): sym for sym in desired
        }

        to_subscribe = desired - self._subscribed_symbols
        to_unsubscribe = self._subscribed_symbols - desired

        if to_unsubscribe:
            await self._ws_unsubscribe(to_unsubscribe)
        if to_subscribe:
            await self._ws_subscribe(to_subscribe)

        self._subscribed_symbols = desired.copy()

        if desired:
            logger.info(f"PriceMonitor watching: {sorted(desired)}")
        else:
            logger.debug("PriceMonitor: no open positions to watch.")

    async def _ws_subscribe(self, symbols: set[str]):
        """Send a SUBSCRIBE request for the given symbols."""
        if not self._ws:
            return

        streams = [_symbol_to_stream(s) for s in symbols]
        payload = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000),
        }
        await self._ws.send(json.dumps(payload))
        logger.debug(f"Subscribed to streams: {streams}")

    async def _ws_unsubscribe(self, symbols: set[str]):
        """Send an UNSUBSCRIBE request for the given symbols."""
        if not self._ws:
            return

        streams = [_symbol_to_stream(s) for s in symbols]
        payload = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000),
        }
        await self._ws.send(json.dumps(payload))
        logger.debug(f"Unsubscribed from streams: {streams}")

    # ── Internal: Message Processing ────────────────────────────

    def _handle_message(self, raw_msg: str):
        """
        Parse an incoming WebSocket message and, if it contains a
        price update for a monitored symbol, run stop/TP checks.

        Binance combined-stream wrapper format:
        {
            "stream": "btcusdt@miniTicker",
            "data": {
                "e": "24hrMiniTicker",
                "s": "BTCUSDT",
                "c": "27000.50",    <-- close / last price
                ...
            }
        }
        """
        msg = json.loads(raw_msg)

        # Subscription confirmations have an "id" field — skip them.
        if "id" in msg and "result" in msg:
            return

        stream = msg.get("stream", "")
        data = msg.get("data")
        if not data or not stream:
            return

        # Extract last price from miniTicker
        price_str = data.get("c")
        if price_str is None:
            return

        price = float(price_str)
        pair_lower = _stream_to_symbol(stream)

        # Resolve original symbol key (e.g. "BTC/USDT")
        symbol = self._pair_to_symbol.get(pair_lower)
        if not symbol:
            return

        # Store latest price
        self.latest_prices[symbol] = price

        # Run stop / take-profit check via the risk manager
        self._check_and_act(symbol, price)

    def _check_and_act(self, symbol: str, price: float):
        """
        Call risk_manager.check_stops_and_tp with the latest price
        for a single symbol.  If any trades are closed, send a
        Telegram alert and update subscriptions.
        """
        with self._lock:
            # check_stops_and_tp expects a dict of {symbol: price}
            closed_trades = self.risk_manager.check_stops_and_tp({symbol: price})

            if not closed_trades:
                return

            for trade in closed_trades:
                sym = trade.get("symbol", symbol)
                pnl_pct = trade.get("pnl_pct", 0)
                pnl_usd = trade.get("pnl_usd", 0)
                reason = trade.get("reason", "unknown")
                capital = self.risk_manager.state.get("capital", 0)

                logger.info(
                    f"[PriceMonitor] Stop triggered for {sym} at ${price:.4f} "
                    f"| reason={reason} | PnL={pnl_pct:+.2f}% (${pnl_usd:+.2f})"
                )

                # Send Telegram alert
                if self.telegram:
                    try:
                        self.telegram.send_trade_closed(
                            symbol=sym,
                            pnl_pct=pnl_pct,
                            pnl_usd=pnl_usd,
                            reason=reason,
                            capital=capital,
                        )
                    except Exception as e:
                        logger.error(f"Failed to send Telegram alert: {e}")

            # Position was closed — update subscriptions so we stop
            # watching symbols that are no longer open.
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._sync_subscriptions(), self._loop
                )
