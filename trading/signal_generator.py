"""
Signal Generator v2 — Expanded Alpha Signals
===============================================
Two strategies running on 5 coins:
  1. Momentum Reversal (relaxed params) — contrarian, fewer but high-quality trades
  2. Volatility Breakout (new) — trend-continuation, more frequent trades

Universe: BTC, ETH, BNB, SOL, XRP

Usage:
    from trading.signal_generator import SignalGenerator
    sg = SignalGenerator()
    signals = sg.generate_all()
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger

from utils.indicators import add_all_indicators
from utils.regime import RegimeDetector
from strategies.multi_timeframe import MultiTimeframeStrategy


# ═══════════════════════════════════════════
# ALPHA CONFIGS — v2 (relaxed + expanded)
# ═══════════════════════════════════════════

ALPHA_CONFIGS = {
    # ── Momentum Reversal (relaxed thresholds) ──
    "BTC/USDT": {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": -8.0,
                    "roc_exit": 3.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
            {
                "name": "volatility_breakout",
                "params": {
                    "atr_multiplier": 1.5,
                    "volume_threshold": 1.5,
                    "holding_periods": 6,
                },
                "stop_loss": 0.04,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    },
    "ETH/USDT": {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": -8.0,
                    "roc_exit": 3.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
            {
                "name": "volatility_breakout",
                "params": {
                    "atr_multiplier": 1.5,
                    "volume_threshold": 1.5,
                    "holding_periods": 6,
                },
                "stop_loss": 0.04,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    },
    "BNB/USDT": {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": -8.0,
                    "roc_exit": 3.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
            {
                "name": "volatility_breakout",
                "params": {
                    "atr_multiplier": 2.0,
                    "volume_threshold": 1.5,
                    "holding_periods": 6,
                },
                "stop_loss": 0.04,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    },
    "SOL/USDT": {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": -10.0,
                    "roc_exit": 3.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
            {
                "name": "volatility_breakout",
                "params": {
                    "atr_multiplier": 1.5,
                    "volume_threshold": 1.5,
                    "holding_periods": 6,
                },
                "stop_loss": 0.04,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    },
    "XRP/USDT": {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": -8.0,
                    "roc_exit": 3.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
            {
                "name": "volatility_breakout",
                "params": {
                    "atr_multiplier": 1.5,
                    "volume_threshold": 1.5,
                    "holding_periods": 6,
                },
                "stop_loss": 0.04,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    },
}


class SignalGenerator:
    """Generates real-time trading signals from Binance data."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.regime_detector = RegimeDetector()
        self.mtf_strategy = MultiTimeframeStrategy()
        logger.info("SignalGenerator v3 initialized (5 coins, 3 strategies, regime + MTF).")

    def fetch_latest(self, symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
        """Fetch latest candles from Binance."""
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                candles,
                columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.drop(columns=["timestamp_ms"])
            df = df.set_index("timestamp")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    # ── Strategy: Momentum Reversal ──

    def _check_momentum_reversal(self, df: pd.DataFrame, params: dict) -> dict:
        """Check momentum reversal signal on latest candle."""
        roc_threshold = params["roc_threshold"]
        roc_exit = params["roc_exit"]
        support_buffer = params["support_buffer"]

        latest = df.iloc[-1]
        roc = latest["roc_10"]
        price_pos = latest["price_position"]
        rsi = latest["rsi"]

        near_support = price_pos < support_buffer + 0.1
        near_resistance = price_pos > 1 - support_buffer - 0.1
        oversold = (roc < roc_threshold) and near_support
        overbought = (roc > abs(roc_threshold)) and near_resistance

        signal = 0
        reason = ""

        if oversold:
            signal = 1
            reason = (
                f"MR BUY: ROC({roc:.1f})<{roc_threshold}, "
                f"near support(pos={price_pos:.2f}), RSI={rsi:.0f}"
            )
        elif overbought:
            signal = -1
            reason = (
                f"MR SHORT: ROC({roc:.1f})>{abs(roc_threshold)}, "
                f"near resistance(pos={price_pos:.2f}), RSI={rsi:.0f}"
            )

        return {"signal": signal, "reason": reason, "strategy": "momentum_reversal"}

    # ── Strategy: Volatility Breakout ──

    def _check_volatility_breakout(self, df: pd.DataFrame, params: dict) -> dict:
        """Check volatility breakout signal on latest candle."""
        atr_mult = params["atr_multiplier"]
        vol_thresh = params["volume_threshold"]

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        upper_band = latest["sma_20"] + atr_mult * latest["atr"]
        lower_band = latest["sma_20"] - atr_mult * latest["atr"]
        vol_spike = latest["volume_ratio"] > vol_thresh

        signal = 0
        reason = ""

        if latest["close"] > upper_band and vol_spike:
            # Confirm: previous candle was below → fresh breakout
            prev_upper = prev["sma_20"] + atr_mult * prev["atr"]
            if prev["close"] <= prev_upper:
                signal = 1
                reason = (
                    f"VB BUY: price ${latest['close']:.2f} > upper "
                    f"${upper_band:.2f}, vol_ratio={latest['volume_ratio']:.1f}x"
                )
        elif latest["close"] < lower_band and vol_spike:
            prev_lower = prev["sma_20"] - atr_mult * prev["atr"]
            if prev["close"] >= prev_lower:
                signal = -1
                reason = (
                    f"VB SHORT: price ${latest['close']:.2f} < lower "
                    f"${lower_band:.2f}, vol_ratio={latest['volume_ratio']:.1f}x"
                )

        return {"signal": signal, "reason": reason, "strategy": "volatility_breakout"}

    # ── Strategy: Multi-Timeframe (1D trend + 4H entry) ──

    def _check_multi_timeframe(self, symbol: str, timeframe: str, df_4h: pd.DataFrame) -> dict:
        """Check multi-timeframe signal using 1D trend + 4H entry."""
        try:
            # Fetch daily data
            df_1d = self.fetch_latest(symbol, "1d", limit=100)
            if df_1d.empty or len(df_1d) < 55:
                return None

            df_1d = add_all_indicators(df_1d)
            result = self.mtf_strategy.check_current_signal(df_4h, df_1d)

            if result["signal"] != 0:
                logger.info(
                    f"  MTF signal: {symbol} daily_trend={result['daily_trend_label']}, "
                    f"signal={result['signal']}"
                )

            return result
        except Exception as e:
            logger.debug(f"MTF check failed for {symbol}: {e}")
            return None

    # ── Main Signal Logic ──

    def generate_signal(self, symbol: str) -> dict:
        """
        Generate signal for a symbol by checking all configured strategies.
        Priority: first strategy that fires wins (Momentum Reversal first).
        """
        if symbol not in ALPHA_CONFIGS:
            return {"signal": 0, "reason": "Not configured", "symbol": symbol}

        config = ALPHA_CONFIGS[symbol]
        df = self.fetch_latest(symbol, config["timeframe"], config["lookback_candles"])

        if df.empty:
            return {"signal": 0, "reason": "No data", "symbol": symbol}

        df = add_all_indicators(df)
        latest = df.iloc[-1]

        # Detect market regime
        regime_info = self.regime_detector.detect(df)
        recommendation = self.regime_detector.get_strategy_recommendation(regime_info)
        avoid_strategies = recommendation["avoid"]

        logger.debug(
            f"{symbol} regime={regime_info['regime']} "
            f"(confidence={regime_info['confidence']:.0%}, ADX={regime_info['adx']:.1f})"
        )

        # Check each strategy in order, skip if regime says avoid
        for strat_config in config["strategies"]:
            name = strat_config["name"]
            params = strat_config["params"]

            if name in avoid_strategies:
                logger.debug(f"  Skipping {name} — avoided in {regime_info['regime']} regime")
                continue

            if name == "momentum_reversal":
                result = self._check_momentum_reversal(df, params)
            elif name == "volatility_breakout":
                result = self._check_volatility_breakout(df, params)
            else:
                continue

            if result["signal"] != 0:
                return {
                    **result,
                    "symbol": symbol,
                    "timeframe": config["timeframe"],
                    "stop_loss": strat_config["stop_loss"],
                    "close": latest["close"],
                    "roc_10": latest["roc_10"],
                    "rsi": latest["rsi"],
                    "price_position": latest["price_position"],
                    "atr_pct": latest["atr_pct"],
                    "bb_pct_b": latest["bb_pct_b"],
                    "volume_ratio": latest["volume_ratio"],
                    "adx": latest["adx"],
                    "regime": regime_info["regime"],
                    "regime_confidence": regime_info["confidence"],
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    ),
                }

        # No signal from primary strategies — try multi-timeframe
        mtf_result = self._check_multi_timeframe(symbol, config["timeframe"], df)
        if mtf_result and mtf_result["signal"] != 0:
            return {
                **mtf_result,
                "symbol": symbol,
                "timeframe": config["timeframe"],
                "stop_loss": 0.03,
                "close": latest["close"],
                "roc_10": latest["roc_10"],
                "rsi": latest["rsi"],
                "price_position": latest["price_position"],
                "atr_pct": latest["atr_pct"],
                "bb_pct_b": latest["bb_pct_b"],
                "volume_ratio": latest["volume_ratio"],
                "adx": latest["adx"],
                "regime": regime_info["regime"],
                "regime_confidence": regime_info["confidence"],
                "timestamp": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
            }

        # No signal at all — check exit logic
        mr_params = config["strategies"][0]["params"]  # First is always MR
        roc = latest["roc_10"]
        exit_reason = ""

        if roc > mr_params["roc_exit"]:
            exit_reason = f"MR exit zone: ROC({roc:.1f}) > {mr_params['roc_exit']}"
        elif roc < -mr_params["roc_exit"]:
            exit_reason = f"MR exit zone: ROC({roc:.1f}) < -{mr_params['roc_exit']}"

        return {
            "signal": 0,
            "reason": exit_reason if exit_reason else "No signal",
            "strategy": "none",
            "symbol": symbol,
            "timeframe": config["timeframe"],
            "stop_loss": config["strategies"][0]["stop_loss"],
            "close": latest["close"],
            "roc_10": roc,
            "rsi": latest["rsi"],
            "price_position": latest["price_position"],
            "atr_pct": latest["atr_pct"],
            "bb_pct_b": latest["bb_pct_b"],
            "volume_ratio": latest["volume_ratio"],
            "adx": latest["adx"],
            "regime": regime_info["regime"],
            "regime_confidence": regime_info["confidence"],
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        }

    def generate_all(self) -> list:
        """Generate signals for all configured symbols."""
        signals = []
        for symbol in ALPHA_CONFIGS:
            sig = self.generate_signal(symbol)
            signals.append(sig)

            if sig["signal"] == 1:
                emoji = "🟢"
            elif sig["signal"] == -1:
                emoji = "🔴"
            else:
                emoji = "⚪"

            logger.info(
                f"{emoji} {symbol}: signal={sig['signal']} "
                f"[{sig.get('strategy','none')}] | {sig['reason']}"
            )

        return signals


if __name__ == "__main__":
    sg = SignalGenerator()
    signals = sg.generate_all()

    print("\n" + "=" * 65)
    print("  CURRENT SIGNALS (v2 — 5 coins, 2 strategies)")
    print("=" * 65)
    for s in signals:
        if s["signal"] == 1:
            emoji = "🟢 BUY"
        elif s["signal"] == -1:
            emoji = "🔴 SHORT"
        else:
            emoji = "⚪ FLAT"

        print(f"\n  {s['symbol']} — {emoji}")
        print(f"    Price: ${s['close']:,.4f} | ROC(10): {s['roc_10']:.2f} | "
              f"RSI: {s['rsi']:.1f} | ADX: {s['adx']:.1f}")
        print(f"    Vol Ratio: {s['volume_ratio']:.2f} | "
              f"Price Pos: {s['price_position']:.2f} | ATR%: {s['atr_pct']:.2f}")
        if s["reason"]:
            print(f"    → {s['reason']}")
    print()