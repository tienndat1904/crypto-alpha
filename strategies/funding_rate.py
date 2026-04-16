"""
Funding Rate Arbitrage Strategy
================================
Uses Binance Futures funding rate as a contrarian signal.

Logic:
  - Funding rate > +0.05% (8h) = market over-leveraged LONG → SHORT signal
  - Funding rate < -0.03% (8h) = market over-leveraged SHORT → LONG signal
  - Track funding rate trend (rising/falling) for confirmation
  - Works best combined with technical signals as a filter or standalone

Data source: Binance USDT-M Futures API via ccxt
"""

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from loguru import logger



class FundingRateStrategy:
    """Contrarian strategy based on futures funding rates."""

    # Thresholds
    HIGH_FUNDING = 0.0005      # +0.05% per 8h → crowd is too long
    LOW_FUNDING = -0.0003      # -0.03% per 8h → crowd is too short
    EXTREME_FUNDING = 0.001    # +0.1% → extreme FOMO
    EXTREME_NEG_FUNDING = -0.0006  # -0.06% → extreme panic

    def __init__(self):
        # Public endpoints only — no API key needed for funding rate data
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 min cache
        logger.info("FundingRateStrategy initialized.")

    def _symbol_to_futures(self, symbol: str) -> str:
        """Convert spot symbol to futures format. e.g. BTC/USDT -> BTC/USDT:USDT"""
        if ":USDT" not in symbol:
            return symbol.replace("/USDT", "/USDT:USDT")
        return symbol

    def fetch_funding_rate(self, symbol: str) -> dict:
        """
        Fetch current and historical funding rate for a symbol.

        Returns:
            dict with:
                - current_rate: latest funding rate (float)
                - avg_rate_24h: average over last 3 periods (24h)
                - avg_rate_3d: average over last 9 periods (3 days)
                - trend: 'rising', 'falling', 'flat'
                - history: list of recent rates
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
            # Fetch funding rate history (last 3 days = ~9 periods)
            since = int((now - timedelta(days=3)).timestamp() * 1000)
            rates = self.exchange.fetch_funding_rate_history(
                futures_symbol, since=since, limit=9
            )

            if not rates:
                return self._default_result()

            rate_values = [r["fundingRate"] for r in rates if r.get("fundingRate") is not None]
            if not rate_values:
                return self._default_result()

            current_rate = rate_values[-1]
            avg_24h = np.mean(rate_values[-3:]) if len(rate_values) >= 3 else current_rate
            avg_3d = np.mean(rate_values)

            # Trend: compare last 3 vs previous 3
            if len(rate_values) >= 6:
                recent = np.mean(rate_values[-3:])
                older = np.mean(rate_values[-6:-3])
                diff = recent - older
                if diff > 0.0001:
                    trend = "rising"
                elif diff < -0.0001:
                    trend = "falling"
                else:
                    trend = "flat"
            else:
                trend = "flat"

            result = {
                "current_rate": current_rate,
                "avg_rate_24h": avg_24h,
                "avg_rate_3d": avg_3d,
                "trend": trend,
                "history": rate_values,
                "available": True,
            }

            self._cache[cache_key] = result
            self._cache_time[cache_key] = now
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
            return self._default_result()

    def _default_result(self):
        return {
            "current_rate": 0.0,
            "avg_rate_24h": 0.0,
            "avg_rate_3d": 0.0,
            "trend": "flat",
            "history": [],
            "available": False,
        }

    def check_signal(self, symbol: str, df: pd.DataFrame = None) -> dict:
        """
        Generate signal based on funding rate.

        Args:
            symbol: e.g. "BTC/USDT"
            df: OHLCV dataframe (optional, used for price context)

        Returns:
            dict with signal, reason, strategy, confidence
        """
        fr = self.fetch_funding_rate(symbol)

        if not fr["available"]:
            return {
                "signal": 0,
                "reason": "Funding rate data unavailable",
                "strategy": "funding_rate",
                "confidence": 0.0,
                "funding_rate": 0.0,
            }

        current = fr["current_rate"]
        avg_24h = fr["avg_rate_24h"]
        trend = fr["trend"]

        signal = 0
        reason = ""
        confidence = 0.0

        # Extreme funding → strong contrarian signal
        if current >= self.EXTREME_FUNDING:
            signal = -1  # SHORT — crowd is extremely long
            confidence = 0.9
            reason = (
                f"FR SHORT: extreme funding {current*100:.4f}% "
                f"(avg24h={avg_24h*100:.4f}%), trend={trend}"
            )
        elif current >= self.HIGH_FUNDING:
            signal = -1  # SHORT — crowd is over-leveraged long
            confidence = 0.7
            reason = (
                f"FR SHORT: high funding {current*100:.4f}% "
                f"(avg24h={avg_24h*100:.4f}%), trend={trend}"
            )
        elif current <= self.EXTREME_NEG_FUNDING:
            signal = 1  # LONG — crowd is extremely short
            confidence = 0.9
            reason = (
                f"FR BUY: extreme neg funding {current*100:.4f}% "
                f"(avg24h={avg_24h*100:.4f}%), trend={trend}"
            )
        elif current <= self.LOW_FUNDING:
            signal = 1  # LONG — crowd is over-leveraged short
            confidence = 0.7
            reason = (
                f"FR BUY: negative funding {current*100:.4f}% "
                f"(avg24h={avg_24h*100:.4f}%), trend={trend}"
            )

        # Trend confirmation boost
        if signal == -1 and trend == "rising":
            confidence = min(1.0, confidence + 0.1)
        elif signal == 1 and trend == "falling":
            confidence = min(1.0, confidence + 0.1)

        # Trend contradiction penalty
        if signal == -1 and trend == "falling":
            confidence *= 0.7
        elif signal == 1 and trend == "rising":
            confidence *= 0.7

        return {
            "signal": signal,
            "reason": reason if reason else f"FR neutral: {current*100:.4f}%",
            "strategy": "funding_rate",
            "confidence": round(confidence, 2),
            "funding_rate": current,
            "funding_avg_24h": avg_24h,
            "funding_trend": trend,
        }

    def get_all_rates(self, symbols: list) -> dict:
        """Fetch funding rates for all symbols. Returns {symbol: rate_info}."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_funding_rate(symbol)
        return results
