"""
Liquidation Cascade Detection Strategy
========================================
Detects liquidation cascades by monitoring Open Interest (OI) changes
combined with sharp price movements.

Logic:
  - OI drops sharply (>5% in 4h) + price dumps → LONG liquidation cascade
    → BUY after cascade exhaustion (contrarian)
  - OI drops sharply + price pumps → SHORT liquidation cascade
    → SHORT after cascade exhaustion
  - OI surges + price stable → position buildup, potential breakout coming

Data: Binance Futures Open Interest via ccxt
"""

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from loguru import logger



class LiquidationCascadeStrategy:
    """Detects liquidation cascades via OI + price action."""

    # Thresholds
    OI_DROP_PCT = -0.05        # -5% OI drop in window → cascade likely
    OI_SURGE_PCT = 0.08        # +8% OI surge → position buildup
    PRICE_MOVE_PCT = 0.03      # 3% price move confirms cascade direction
    CASCADE_COOLDOWN = 4       # Hours after cascade before entering

    def __init__(self):
        # Public endpoints only — no API key needed for OI data
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 min cache
        logger.info("LiquidationCascadeStrategy initialized.")

    def _symbol_to_futures(self, symbol: str) -> str:
        if ":USDT" not in symbol:
            return symbol.replace("/USDT", "/USDT:USDT")
        return symbol

    def fetch_open_interest_history(self, symbol: str, periods: int = 12) -> dict:
        """
        Fetch open interest data and compute changes.

        Args:
            symbol: e.g. "BTC/USDT"
            periods: number of 4h periods to fetch (default 12 = 48h)

        Returns:
            dict with OI metrics
        """
        now = datetime.now(timezone.utc)
        cache_key = symbol
        if (
            cache_key in self._cache
            and (now - self._cache_time.get(cache_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() < self._cache_ttl
        ):
            return self._cache[cache_key]

        futures_symbol = self._symbol_to_futures(symbol)

        try:
            # Fetch current OI
            oi_data = self.exchange.fetch_open_interest(futures_symbol)
            current_oi = oi_data.get("openInterestAmount", 0)
            current_oi_value = oi_data.get("openInterestValue", 0)

            if current_oi == 0:
                return self._default_result()

            # Fetch OI history via OHLCV on futures (as proxy for OI trend)
            # Use funding + OI endpoint
            since = int((now - timedelta(hours=4 * periods)).timestamp() * 1000)

            oi_history = self.exchange.fetch_open_interest_history(
                futures_symbol, timeframe="4h", since=since, limit=periods
            )

            if not oi_history or len(oi_history) < 3:
                return self._default_result()

            oi_values = [h["openInterestValue"] for h in oi_history if h.get("openInterestValue")]
            if len(oi_values) < 3:
                return self._default_result()

            # Compute OI changes
            oi_arr = np.array(oi_values)
            oi_pct_changes = np.diff(oi_arr) / oi_arr[:-1]

            # Recent change (last 4h)
            recent_change = oi_pct_changes[-1] if len(oi_pct_changes) > 0 else 0
            # 12h change (last 3 periods)
            change_12h = (oi_arr[-1] - oi_arr[-4]) / oi_arr[-4] if len(oi_arr) >= 4 else 0
            # 24h change
            change_24h = (oi_arr[-1] - oi_arr[-7]) / oi_arr[-7] if len(oi_arr) >= 7 else 0

            # Detect sharp OI drop (cascade signature)
            max_drop = np.min(oi_pct_changes[-3:]) if len(oi_pct_changes) >= 3 else 0
            max_surge = np.max(oi_pct_changes[-3:]) if len(oi_pct_changes) >= 3 else 0

            # OI trend
            if len(oi_arr) >= 6:
                recent_avg = np.mean(oi_arr[-3:])
                older_avg = np.mean(oi_arr[-6:-3])
                if recent_avg > older_avg * 1.03:
                    oi_trend = "rising"
                elif recent_avg < older_avg * 0.97:
                    oi_trend = "falling"
                else:
                    oi_trend = "flat"
            else:
                oi_trend = "flat"

            result = {
                "current_oi": current_oi,
                "current_oi_value": current_oi_value,
                "recent_change_pct": recent_change,
                "change_12h_pct": change_12h,
                "change_24h_pct": change_24h,
                "max_drop_4h": max_drop,
                "max_surge_4h": max_surge,
                "oi_trend": oi_trend,
                "oi_history": oi_values,
                "available": True,
            }

            self._cache[cache_key] = result
            self._cache_time[cache_key] = now
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch OI for {symbol}: {e}")
            return self._default_result()

    def _default_result(self):
        return {
            "current_oi": 0,
            "current_oi_value": 0,
            "recent_change_pct": 0,
            "change_12h_pct": 0,
            "change_24h_pct": 0,
            "max_drop_4h": 0,
            "max_surge_4h": 0,
            "oi_trend": "flat",
            "oi_history": [],
            "available": False,
        }

    def check_signal(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Generate signal based on OI + price cascade detection.

        Args:
            symbol: e.g. "BTC/USDT"
            df: OHLCV dataframe with recent candles

        Returns:
            dict with signal, reason, strategy, confidence
        """
        oi = self.fetch_open_interest_history(symbol)

        if not oi["available"] or df.empty or len(df) < 6:
            return {
                "signal": 0,
                "reason": "OI data unavailable",
                "strategy": "liquidation_cascade",
                "confidence": 0.0,
            }

        # Price changes
        latest_close = df["close"].iloc[-1]
        close_4h_ago = df["close"].iloc[-2] if len(df) >= 2 else latest_close
        close_12h_ago = df["close"].iloc[-4] if len(df) >= 4 else latest_close

        price_change_4h = (latest_close - close_4h_ago) / close_4h_ago
        price_change_12h = (latest_close - close_12h_ago) / close_12h_ago

        oi_change = oi["change_12h_pct"]
        max_drop = oi["max_drop_4h"]
        max_surge = oi["max_surge_4h"]
        oi_trend = oi["oi_trend"]

        signal = 0
        reason = ""
        confidence = 0.0

        # ── Pattern 1: Long Liquidation Cascade ──
        # OI drops sharply + price drops = longs got liquidated
        # → BUY after exhaustion (contrarian)
        if max_drop <= self.OI_DROP_PCT and price_change_12h < -self.PRICE_MOVE_PCT:
            signal = 1
            confidence = 0.75
            reason = (
                f"LC BUY: long cascade detected — "
                f"OI drop {max_drop*100:+.1f}%, "
                f"price {price_change_12h*100:+.1f}% (12h), "
                f"buying exhaustion"
            )

        # ── Pattern 2: Short Liquidation Cascade ──
        # OI drops sharply + price pumps = shorts got liquidated
        # → SHORT after exhaustion
        elif max_drop <= self.OI_DROP_PCT and price_change_12h > self.PRICE_MOVE_PCT:
            signal = -1
            confidence = 0.75
            reason = (
                f"LC SHORT: short squeeze detected — "
                f"OI drop {max_drop*100:+.1f}%, "
                f"price {price_change_12h*100:+.1f}% (12h), "
                f"shorting after squeeze"
            )

        # ── Pattern 3: OI Buildup → Breakout Coming ──
        # OI surging + price flat = big move coming
        # Direction: follow the eventual breakout
        elif max_surge >= self.OI_SURGE_PCT and abs(price_change_12h) < 0.015:
            # OI building up but price hasn't moved — anticipate breakout
            # Use RSI for direction hint
            if len(df) > 0 and "rsi" in df.columns:
                rsi = df["rsi"].iloc[-1]
                if rsi > 55:
                    signal = 1
                    confidence = 0.5
                    reason = (
                        f"LC BUILDUP BUY: OI surge {max_surge*100:+.1f}%, "
                        f"price flat, RSI={rsi:.0f} bullish bias"
                    )
                elif rsi < 45:
                    signal = -1
                    confidence = 0.5
                    reason = (
                        f"LC BUILDUP SHORT: OI surge {max_surge*100:+.1f}%, "
                        f"price flat, RSI={rsi:.0f} bearish bias"
                    )

        # Confidence adjustments
        if signal != 0:
            # Stronger if OI drop is extreme
            if abs(max_drop) > 0.08:
                confidence = min(1.0, confidence + 0.15)

            # Weaker if OI trend contradicts
            if signal == 1 and oi_trend == "rising":
                confidence *= 0.8  # Longs still entering, cascade may not be over
            elif signal == -1 and oi_trend == "falling":
                confidence *= 0.8

        return {
            "signal": signal,
            "reason": reason if reason else f"LC neutral: OI change {oi_change*100:+.1f}%",
            "strategy": "liquidation_cascade",
            "confidence": round(confidence, 2),
            "oi_change_12h": round(oi_change * 100, 2),
            "oi_trend": oi_trend,
            "price_change_12h": round(price_change_12h * 100, 2),
        }
