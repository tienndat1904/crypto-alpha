"""
On-Chain Alpha Strategies
==========================
Uses on-chain data as FILTERS to enhance existing technical strategies.

Logic: On-chain signals (daily) confirm/deny technical signals (4H).
  - Extreme Fear + technical buy → STRONG BUY (higher confidence)
  - Extreme Greed + technical buy → SKIP (on-chain says overbought)
  - Neutral on-chain → let technical signal decide

Strategies:
  1. Fear & Greed Filter — overlay F&G on MR/VB signals
  2. Volume Divergence — on-chain volume vs price divergence
  3. ETH/BTC Rotation — alt season detection

Usage:
    from strategies.onchain_alphas import OnchainSignalFilter
    filter = OnchainSignalFilter()
    enhanced = filter.apply(technical_signal, symbol)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from loguru import logger

from data.onchain.fetcher import OnchainFetcher


class OnchainSignalFilter:
    """
    Enhances technical trading signals with on-chain data.
    Acts as a confidence multiplier, not a standalone signal generator.
    """

    def __init__(self):
        self.fetcher = OnchainFetcher()
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 3600  # Refresh on-chain data every 1 hour
        logger.info("OnchainSignalFilter initialized.")

    def _get_onchain_data(self) -> pd.DataFrame:
        """Get on-chain data with caching."""
        now = datetime.now(timezone.utc)
        if (
            self._cache_time is None
            or (now - self._cache_time).total_seconds() > self._cache_ttl
            or not len(self._cache)
        ):
            self._cache = self.fetcher.fetch_all(days=90)
            self._cache_time = now
            logger.info("On-chain data cache refreshed.")
        return self._cache

    def get_current_regime(self) -> dict:
        """
        Determine current market regime from on-chain data.

        Returns:
            dict with regime info:
                - regime: 'fear', 'neutral', 'greed'
                - fng_value: current Fear & Greed (0-100)
                - fng_trend: 'rising', 'falling', 'flat'
                - confidence_multiplier: 0.5 to 1.5
                - should_trade: True/False
                - reason: explanation
        """
        df = self._get_onchain_data()
        if df.empty:
            return {
                "regime": "unknown",
                "fng_value": 50,
                "confidence_multiplier": 1.0,
                "should_trade": True,
                "reason": "On-chain data unavailable, using default",
            }

        latest = df.iloc[-1]
        fng = latest.get("fng_value", 50)
        fng_sma7 = latest.get("fng_sma_7", 50)
        fng_momentum = latest.get("fng_momentum", 0)

        # Determine regime
        if fng < 25:
            regime = "extreme_fear"
        elif fng < 40:
            regime = "fear"
        elif fng > 75:
            regime = "extreme_greed"
        elif fng > 60:
            regime = "greed"
        else:
            regime = "neutral"

        # F&G trend
        if fng_momentum > 10:
            fng_trend = "rising"
        elif fng_momentum < -10:
            fng_trend = "falling"
        else:
            fng_trend = "flat"

        # Confidence multiplier for signals
        # Extreme fear + buy signal = high confidence (1.5x)
        # Extreme greed + buy signal = low confidence (0.5x)
        if regime == "extreme_fear":
            buy_multiplier = 1.5
            sell_multiplier = 0.5
        elif regime == "fear":
            buy_multiplier = 1.3
            sell_multiplier = 0.7
        elif regime == "extreme_greed":
            buy_multiplier = 0.5
            sell_multiplier = 1.5
        elif regime == "greed":
            buy_multiplier = 0.7
            sell_multiplier = 1.3
        else:
            buy_multiplier = 1.0
            sell_multiplier = 1.0

        # ETH/BTC ratio analysis (alt season)
        eth_btc_change = latest.get("eth_btc_ratio_change", 0)
        alt_season = False
        if isinstance(eth_btc_change, (int, float)) and not np.isnan(eth_btc_change):
            alt_season = eth_btc_change > 0.05  # ETH outperforming BTC by 5%+ in a week

        return {
            "regime": regime,
            "fng_value": int(fng),
            "fng_sma_7": round(fng_sma7, 1) if not np.isnan(fng_sma7) else 50,
            "fng_trend": fng_trend,
            "fng_momentum": round(fng_momentum, 1) if not np.isnan(fng_momentum) else 0,
            "buy_multiplier": buy_multiplier,
            "sell_multiplier": sell_multiplier,
            "alt_season": alt_season,
            "should_trade": True,
            "reason": f"F&G={int(fng)} ({regime}), trend={fng_trend}",
        }

    def enhance_signal(self, signal: int, symbol: str) -> dict:
        """
        Enhance a technical signal with on-chain context.

        Args:
            signal: 1 (buy), -1 (short), 0 (flat)
            symbol: Trading pair, e.g. "BTC/USDT"

        Returns:
            dict:
                - enhanced_signal: adjusted signal (may be blocked)
                - confidence: 0.0 to 1.0
                - regime: current market regime
                - reason: explanation
        """
        regime = self.get_current_regime()

        if signal == 0:
            return {
                "enhanced_signal": 0,
                "confidence": 0.0,
                "regime": regime["regime"],
                "reason": "No technical signal",
            }

        # Apply confidence multiplier
        if signal == 1:  # Buy
            multiplier = regime["buy_multiplier"]
        else:  # Short
            multiplier = regime["sell_multiplier"]

        confidence = min(1.0, 0.5 * multiplier)

        # Block signal if confidence too low
        if confidence < 0.3:
            return {
                "enhanced_signal": 0,
                "confidence": confidence,
                "regime": regime["regime"],
                "reason": (
                    f"Signal BLOCKED by on-chain filter: "
                    f"F&G={regime['fng_value']} ({regime['regime']}) "
                    f"conflicts with {'BUY' if signal == 1 else 'SHORT'} signal"
                ),
            }

        # Alt season check for non-BTC coins
        if regime["alt_season"] and "BTC" not in symbol and signal == 1:
            confidence = min(1.0, confidence * 1.2)

        return {
            "enhanced_signal": signal,
            "confidence": round(confidence, 2),
            "regime": regime["regime"],
            "fng_value": regime["fng_value"],
            "fng_trend": regime["fng_trend"],
            "buy_multiplier": regime["buy_multiplier"],
            "sell_multiplier": regime["sell_multiplier"],
            "reason": (
                f"Signal CONFIRMED: F&G={regime['fng_value']} ({regime['regime']}), "
                f"confidence={confidence:.0%}"
            ),
        }


# ═══════════════════════════════════════════
# BACKTEST HELPER
# ═══════════════════════════════════════════

def add_onchain_to_ohlcv(
    ohlcv_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily on-chain features into intraday OHLCV data.
    Each 4H candle gets the on-chain value from that day.

    Args:
        ohlcv_df: OHLCV with datetime index (e.g. 4H)
        onchain_df: On-chain features with datetime index (daily)

    Returns:
        Merged DataFrame
    """
    ohlcv = ohlcv_df.copy()
    ohlcv["date"] = ohlcv.index.date

    onchain = onchain_df.copy()
    if "date" not in onchain.columns:
        onchain["date"] = onchain.index.date

    # Select on-chain columns to merge
    onchain_cols = [
        "date", "fng_value", "fng_sma_7", "fng_sma_30",
        "fng_extreme_fear", "fng_extreme_greed", "fng_momentum", "fng_zscore",
    ]
    available = [c for c in onchain_cols if c in onchain.columns]
    onchain_merge = onchain[available].drop_duplicates(subset=["date"], keep="last")

    merged = ohlcv.merge(onchain_merge, on="date", how="left")
    merged = merged.drop(columns=["date"])

    # Forward fill (on-chain data updates daily, 4H candles need fill)
    onchain_fill_cols = [c for c in available if c != "date"]
    merged[onchain_fill_cols] = merged[onchain_fill_cols].ffill()

    logger.info(f"Merged {len(onchain_fill_cols)} on-chain features into OHLCV ({len(merged)} rows)")
    return merged


if __name__ == "__main__":
    filter = OnchainSignalFilter()

    print("\n=== Current Market Regime ===")
    regime = filter.get_current_regime()
    for k, v in regime.items():
        print(f"  {k}: {v}")

    print("\n=== Signal Enhancement Examples ===")
    for signal, symbol in [(1, "BTC/USDT"), (-1, "BTC/USDT"), (1, "SOL/USDT")]:
        result = filter.enhance_signal(signal, symbol)
        direction = "BUY" if signal == 1 else "SHORT"
        print(f"\n  {direction} {symbol}:")
        print(f"    Enhanced: {result['enhanced_signal']}")
        print(f"    Confidence: {result['confidence']:.0%}")
        print(f"    {result['reason']}")
    print()
