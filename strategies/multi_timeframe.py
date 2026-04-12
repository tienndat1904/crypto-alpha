"""
Multi-Timeframe Strategy
=========================
Uses daily (1D) data to determine the major trend direction,
then uses 4H data to find precise entry points.

Logic:
  1D Trend Filter:
    - EMA_21 > EMA_50 AND price > EMA_21 → Bullish bias (only long)
    - EMA_21 < EMA_50 AND price < EMA_21 → Bearish bias (only short)
    - Otherwise → Neutral (no trade)

  4H Entry (when 1D confirms direction):
    - Bullish: RSI < 40 pullback OR breakout above BB upper with volume
    - Bearish: RSI > 60 rally OR breakdown below BB lower with volume

  4H Exit:
    - Bullish: RSI > 70 OR price crosses below EMA_21
    - Bearish: RSI < 30 OR price crosses above EMA_21

This filters out counter-trend trades that cause most losses
in momentum/breakout strategies.

Usage:
    from strategies.multi_timeframe import MultiTimeframeStrategy
    mtf = MultiTimeframeStrategy()
    signals = mtf.generate_signals(df_4h, df_1d)
"""

import pandas as pd
import numpy as np
from loguru import logger


class MultiTimeframeStrategy:
    """
    Combines 1D trend direction with 4H entry timing.
    """

    def __init__(
        self,
        # 1D trend params
        fast_ema: int = 21,
        slow_ema: int = 50,
        # 4H entry params
        rsi_pullback_long: float = 40,
        rsi_pullback_short: float = 60,
        rsi_exit_long: float = 70,
        rsi_exit_short: float = 30,
        volume_threshold: float = 1.3,
        # BB breakout
        bb_entry_threshold: float = 0.95,
        bb_entry_threshold_short: float = 0.05,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_pullback_long = rsi_pullback_long
        self.rsi_pullback_short = rsi_pullback_short
        self.rsi_exit_long = rsi_exit_long
        self.rsi_exit_short = rsi_exit_short
        self.volume_threshold = volume_threshold
        self.bb_entry_threshold = bb_entry_threshold
        self.bb_entry_threshold_short = bb_entry_threshold_short

    def get_daily_trend(self, df_1d: pd.DataFrame) -> pd.Series:
        """
        Determine daily trend bias.

        Returns:
            Series with 1 (bullish), -1 (bearish), 0 (neutral)
            indexed by date.
        """
        ema_fast_col = f"ema_{self.fast_ema}"
        ema_slow_col = f"ema_{self.slow_ema}"

        # Use existing EMA columns or compute them
        if ema_fast_col not in df_1d.columns:
            df_1d = df_1d.copy()
            df_1d[ema_fast_col] = df_1d["close"].ewm(span=self.fast_ema, adjust=False).mean()
        if ema_slow_col not in df_1d.columns:
            df_1d = df_1d.copy()
            df_1d[ema_slow_col] = df_1d["close"].ewm(span=self.slow_ema, adjust=False).mean()

        trend = pd.Series(0, index=df_1d.index, dtype=int)

        bullish = (
            (df_1d[ema_fast_col] > df_1d[ema_slow_col]) &
            (df_1d["close"] > df_1d[ema_fast_col])
        )
        bearish = (
            (df_1d[ema_fast_col] < df_1d[ema_slow_col]) &
            (df_1d["close"] < df_1d[ema_fast_col])
        )

        trend[bullish] = 1
        trend[bearish] = -1

        return trend

    def merge_daily_trend_to_4h(
        self,
        df_4h: pd.DataFrame,
        daily_trend: pd.Series,
    ) -> pd.Series:
        """
        Forward-fill daily trend into 4H timeframe.
        Each 4H bar gets the trend from its corresponding day.
        """
        # Normalize daily trend index to date only
        daily_trend_dated = daily_trend.copy()
        daily_trend_dated.index = daily_trend_dated.index.normalize()

        # Map 4H timestamps to their date
        dates_4h = df_4h.index.normalize()
        trend_4h = dates_4h.map(
            lambda d: daily_trend_dated.get(d, np.nan)
        )
        trend_4h = pd.Series(trend_4h, index=df_4h.index).ffill().fillna(0).astype(int)

        return trend_4h

    def generate_signals(
        self,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate trading signals using multi-timeframe logic.

        Args:
            df_4h: 4H OHLCV with indicators (rsi, bb_pct_b, ema_21, volume_ratio)
            df_1d: 1D OHLCV with indicators (or at least close prices)

        Returns:
            Signal Series: 1=long, -1=short, 0=flat
        """
        # Step 1: Get daily trend
        daily_trend = self.get_daily_trend(df_1d)

        # Step 2: Merge to 4H
        trend_4h = self.merge_daily_trend_to_4h(df_4h, daily_trend)

        # Step 3: Generate 4H signals filtered by daily trend
        signal = pd.Series(0, index=df_4h.index, dtype=int)
        position = 0

        for i in range(1, len(df_4h)):
            bias = trend_4h.iloc[i]
            row = df_4h.iloc[i]
            rsi = row.get("rsi", 50)
            bb_pct = row.get("bb_pct_b", 0.5)
            vol_ratio = row.get("volume_ratio", 1.0)
            close = row["close"]
            ema_21 = row.get("ema_21", close)

            # ── Long entries (only when 1D bullish) ──
            if bias == 1 and position <= 0:
                # Pullback entry: RSI dipped then price still above EMA
                pullback = rsi < self.rsi_pullback_long and close > ema_21
                # Breakout entry: price breaks above BB with volume
                breakout = bb_pct > self.bb_entry_threshold and vol_ratio > self.volume_threshold

                if pullback or breakout:
                    position = 1

            # ── Short entries (only when 1D bearish) ──
            elif bias == -1 and position >= 0:
                pullback = rsi > self.rsi_pullback_short and close < ema_21
                breakdown = bb_pct < self.bb_entry_threshold_short and vol_ratio > self.volume_threshold

                if pullback or breakdown:
                    position = -1

            # ── Exit logic ──
            elif position == 1:
                # Exit long: RSI overbought or price drops below EMA
                if rsi > self.rsi_exit_long or close < ema_21:
                    position = 0
                # Also exit if daily trend flips bearish
                if bias == -1:
                    position = 0

            elif position == -1:
                if rsi < self.rsi_exit_short or close > ema_21:
                    position = 0
                if bias == 1:
                    position = 0

            signal.iloc[i] = position

        return signal

    def check_current_signal(
        self,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
    ) -> dict:
        """
        Check signal on the latest bar only (for live trading).

        Returns:
            dict with signal, reason, trend_bias, and metrics
        """
        daily_trend = self.get_daily_trend(df_1d)
        trend_4h = self.merge_daily_trend_to_4h(df_4h, daily_trend)

        latest = df_4h.iloc[-1]
        bias = trend_4h.iloc[-1]
        rsi = latest.get("rsi", 50)
        bb_pct = latest.get("bb_pct_b", 0.5)
        vol_ratio = latest.get("volume_ratio", 1.0)
        close = latest["close"]
        ema_21 = latest.get("ema_21", close)

        bias_label = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[bias]
        signal = 0
        reason = f"1D trend: {bias_label}"

        if bias == 1:
            pullback = rsi < self.rsi_pullback_long and close > ema_21
            breakout = bb_pct > self.bb_entry_threshold and vol_ratio > self.volume_threshold

            if pullback:
                signal = 1
                reason = f"MTF BUY: 1D bullish + 4H RSI pullback ({rsi:.0f})"
            elif breakout:
                signal = 1
                reason = f"MTF BUY: 1D bullish + 4H BB breakout (vol={vol_ratio:.1f}x)"

        elif bias == -1:
            pullback = rsi > self.rsi_pullback_short and close < ema_21
            breakdown = bb_pct < self.bb_entry_threshold_short and vol_ratio > self.volume_threshold

            if pullback:
                signal = -1
                reason = f"MTF SHORT: 1D bearish + 4H RSI rally ({rsi:.0f})"
            elif breakdown:
                signal = -1
                reason = f"MTF SHORT: 1D bearish + 4H BB breakdown (vol={vol_ratio:.1f}x)"

        return {
            "signal": signal,
            "reason": reason,
            "strategy": "multi_timeframe",
            "daily_trend": bias,
            "daily_trend_label": bias_label,
            "rsi": rsi,
            "bb_pct_b": bb_pct,
            "volume_ratio": vol_ratio,
        }
