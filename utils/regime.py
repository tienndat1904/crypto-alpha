"""
Market Regime Detection
========================
Classifies current market state as trending or sideways using ADX and
volatility metrics. Used to select the right strategy:
  - Trending regime  → Trend-Following, Volatility Breakout
  - Sideways regime  → Mean-Reversion, Momentum Reversal

Approach:
  1. ADX > threshold → trending (strong directional movement)
  2. ADX < threshold AND low volatility → sideways (range-bound)
  3. High volatility + low ADX → choppy (avoid trading or reduce size)

Usage:
    from utils.regime import RegimeDetector
    rd = RegimeDetector()
    regime = rd.detect(df)
    # regime = {"regime": "trending", "confidence": 0.85, "adx": 32.1, ...}
"""

import numpy as np
import pandas as pd
from loguru import logger


class RegimeDetector:
    """
    Detects market regime from OHLCV data with indicators.
    Expects DataFrame with 'adx', 'atr_pct', 'bb_width', 'volatility_30d' columns.
    """

    # Regime types
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    CHOPPY = "choppy"

    def __init__(
        self,
        adx_trending: float = 25.0,
        adx_strong_trend: float = 40.0,
        adx_weak: float = 20.0,
        volatility_high_pct: float = 75.0,
        lookback: int = 20,
    ):
        """
        Args:
            adx_trending: ADX above this = trending
            adx_strong_trend: ADX above this = strong trend
            adx_weak: ADX below this = no trend
            volatility_high_pct: Percentile threshold for "high volatility"
            lookback: Bars to consider for regime stability
        """
        self.adx_trending = adx_trending
        self.adx_strong_trend = adx_strong_trend
        self.adx_weak = adx_weak
        self.volatility_high_pct = volatility_high_pct
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect current market regime from latest data.

        Args:
            df: DataFrame with indicators (needs 'adx', 'atr_pct', 'bb_width')

        Returns:
            dict with regime, confidence, and supporting metrics
        """
        if len(df) < self.lookback:
            return self._result(self.SIDEWAYS, 0.5, df)

        recent = df.tail(self.lookback)
        latest = df.iloc[-1]

        adx = latest["adx"]
        adx_mean = recent["adx"].mean()

        # Volatility metrics
        atr_pct = latest.get("atr_pct", 0)
        bb_width = latest.get("bb_width", 0)
        vol_30d = latest.get("volatility_30d", 0)

        # Historical volatility percentile
        if "atr_pct" in df.columns and len(df) > 50:
            vol_percentile = (df["atr_pct"] < atr_pct).mean() * 100
        else:
            vol_percentile = 50.0

        # ADX trend: is it rising or falling?
        adx_rising = recent["adx"].iloc[-1] > recent["adx"].iloc[0]

        # ── Classification ──
        if adx >= self.adx_strong_trend:
            regime = self.TRENDING
            confidence = min(0.95, 0.7 + (adx - self.adx_strong_trend) / 40)
        elif adx >= self.adx_trending:
            regime = self.TRENDING
            confidence = 0.5 + (adx - self.adx_trending) / (self.adx_strong_trend - self.adx_trending) * 0.2
            # Boost if ADX is rising
            if adx_rising:
                confidence += 0.1
        elif adx <= self.adx_weak:
            if vol_percentile > self.volatility_high_pct:
                regime = self.CHOPPY
                confidence = 0.5 + (self.volatility_high_pct - vol_percentile) / 100
            else:
                regime = self.SIDEWAYS
                confidence = 0.5 + (self.adx_weak - adx) / self.adx_weak * 0.3
        else:
            # Ambiguous zone (adx_weak < ADX < adx_trending)
            if adx_rising and vol_percentile < 50:
                regime = self.TRENDING
                confidence = 0.4
            elif not adx_rising and vol_percentile > 60:
                regime = self.CHOPPY
                confidence = 0.4
            else:
                regime = self.SIDEWAYS
                confidence = 0.4

        confidence = max(0.1, min(0.95, confidence))

        return {
            "regime": regime,
            "confidence": confidence,
            "adx": adx,
            "adx_mean": adx_mean,
            "adx_rising": adx_rising,
            "atr_pct": atr_pct,
            "bb_width": bb_width,
            "vol_percentile": vol_percentile,
        }

    def detect_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for each row in the DataFrame (for backtesting).
        Returns a Series with regime labels aligned to df index.
        """
        regimes = pd.Series(self.SIDEWAYS, index=df.index, dtype=str)

        for i in range(self.lookback, len(df)):
            window = df.iloc[max(0, i - self.lookback):i + 1]
            result = self.detect(window)
            regimes.iloc[i] = result["regime"]

        return regimes

    def get_strategy_recommendation(self, regime_info: dict) -> dict:
        """
        Recommend which strategies to use based on detected regime.

        Returns:
            dict with strategy weights and trading advice
        """
        regime = regime_info["regime"]
        confidence = regime_info["confidence"]

        if regime == self.TRENDING:
            return {
                "preferred": ["trend_following", "volatility_breakout"],
                "avoid": ["mean_reversion"],
                "weights": {
                    "mean_reversion": 0.05,
                    "volatility_breakout": 0.35,
                    "trend_following": 0.35,
                    "momentum_reversal": 0.15,
                    "composite": 0.10,
                },
                "position_size_mult": 1.0 if confidence > 0.6 else 0.7,
                "advice": "Trend detected — favor breakout and trend-following",
            }
        elif regime == self.SIDEWAYS:
            return {
                "preferred": ["mean_reversion", "momentum_reversal"],
                "avoid": ["trend_following"],
                "weights": {
                    "mean_reversion": 0.35,
                    "volatility_breakout": 0.05,
                    "trend_following": 0.05,
                    "momentum_reversal": 0.35,
                    "composite": 0.20,
                },
                "position_size_mult": 1.0 if confidence > 0.6 else 0.7,
                "advice": "Range-bound — favor mean-reversion and contrarian",
            }
        else:  # CHOPPY
            return {
                "preferred": [],
                "avoid": ["trend_following", "mean_reversion"],
                "weights": {
                    "mean_reversion": 0.10,
                    "volatility_breakout": 0.10,
                    "trend_following": 0.10,
                    "momentum_reversal": 0.10,
                    "composite": 0.10,
                },
                "position_size_mult": 0.5,
                "advice": "Choppy market — reduce exposure, wait for clarity",
            }

    def _result(self, regime, confidence, df):
        latest = df.iloc[-1] if len(df) > 0 else {}
        return {
            "regime": regime,
            "confidence": confidence,
            "adx": latest.get("adx", 0),
            "adx_mean": 0,
            "adx_rising": False,
            "atr_pct": latest.get("atr_pct", 0),
            "bb_width": latest.get("bb_width", 0),
            "vol_percentile": 50,
        }
