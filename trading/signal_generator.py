"""
Signal Generator v4 — Full Alpha Suite
========================================
Six strategies running on 10 coins:
  1. Momentum Reversal — contrarian on ROC extremes
  2. Volatility Breakout — trend-continuation on ATR bands
  3. Multi-Timeframe — 1D trend + 4H entry
  4. Funding Rate — contrarian on extreme funding rates
  5. Liquidation Cascade — OI drops + price cascades
  6. ML Signal — LightGBM classifier on 20+ features

Universe: BTC, ETH, BNB, SOL, XRP, NEAR, LINK, INJ, UNI, LTC

Usage:
    from trading.signal_generator import SignalGenerator
    sg = SignalGenerator()
    signals = sg.generate_all()
"""

import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from loguru import logger

from utils.indicators import add_all_indicators
from utils.regime import RegimeDetector
from strategies.multi_timeframe import MultiTimeframeStrategy
from strategies.funding_rate import FundingRateStrategy
from strategies.liquidation_cascade import LiquidationCascadeStrategy
from strategies.ml_model import MLSignalModel


# ═══════════════════════════════════════════
# ALPHA CONFIGS — v2 (relaxed + expanded)
# ═══════════════════════════════════════════

def _build_spot_config(roc_thresh=-4.0):
    """Helper to build spot config — Momentum Reversal only (best Sharpe in backtest)."""
    return {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": roc_thresh,
                    "roc_exit": 2.0,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.03,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    }


ALPHA_CONFIGS = {
    # Large caps — lower sensitivity
    "BTC/USDT": _build_spot_config(-4.0),
    "ETH/USDT": _build_spot_config(-4.0),
    "BNB/USDT": _build_spot_config(-4.0),
    "XRP/USDT": _build_spot_config(-4.0),
    "LTC/USDT": _build_spot_config(-4.0),
    # Mid caps
    "LINK/USDT": _build_spot_config(-4.0),
    "UNI/USDT": _build_spot_config(-4.0),
    # Higher volatility — more sensitive threshold
    "SOL/USDT": _build_spot_config(-5.0),
    "NEAR/USDT": _build_spot_config(-5.0),
    "INJ/USDT": _build_spot_config(-5.0),
}


# ═══════════════════════════════════════════
# FUTURES ALPHA CONFIGS — aggressive for leverage
# ═══════════════════════════════════════════
# Key differences vs spot:
#   - Lower ROC threshold → easier entry (leverage amplifies small moves)
#   - Tighter stop-loss → 2% instead of 3% (3x leverage = 6% loss on margin)
#   - Faster exit → roc_exit 1.5 instead of 2.0
#   - VB: lower atr_multiplier → catch breakouts earlier

def _build_futures_config(roc_thresh):
    """Helper to build a futures config for one coin.
    Only Momentum Reversal kept — VB and TF had negative Sharpe in backtest.
    """
    return {
        "strategies": [
            {
                "name": "momentum_reversal",
                "params": {
                    "roc_threshold": roc_thresh,
                    "roc_exit": 1.5,
                    "support_buffer": 0.05,
                },
                "stop_loss": 0.02,
            },
        ],
        "timeframe": "4h",
        "lookback_candles": 250,
    }


FUTURES_ALPHA_CONFIGS = {
    # Large caps — lower threshold (less volatile)
    "BTC/USDT":  _build_futures_config(roc_thresh=-3.0),
    "ETH/USDT":  _build_futures_config(roc_thresh=-3.0),
    "BNB/USDT":  _build_futures_config(roc_thresh=-3.0),
    "XRP/USDT":  _build_futures_config(roc_thresh=-3.0),
    "LTC/USDT":  _build_futures_config(roc_thresh=-3.0),
    # Mid caps — slightly wider (more volatile)
    "SOL/USDT":  _build_futures_config(roc_thresh=-4.0),
    "NEAR/USDT": _build_futures_config(roc_thresh=-4.0),
    "LINK/USDT": _build_futures_config(roc_thresh=-3.0),
    "INJ/USDT":  _build_futures_config(roc_thresh=-4.0),
    "UNI/USDT":  _build_futures_config(roc_thresh=-3.0),
}


class SignalGenerator:
    """Generates real-time trading signals from Binance data."""

    # Cache TTL for multi-timeframe bias (seconds)
    MTF_CACHE_TTL = 300  # 5 minutes

    def __init__(self, configs=None):
        self.configs = configs or ALPHA_CONFIGS
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.regime_detector = RegimeDetector()
        self.mtf_strategy = MultiTimeframeStrategy()
        self._mtf_cache: dict = {}  # {symbol: {"data": dict, "ts": float}}

        # New strategies (shared across all coins)
        self.funding_rate_strategy = FundingRateStrategy()
        self.liquidation_strategy = LiquidationCascadeStrategy()
        self.ml_model = MLSignalModel()

        n_strats = 3 + (1 if self.ml_model.model else 0) + 2  # MR+VB+MTF + ML? + FR+LC
        logger.info(f"SignalGenerator v4 initialized ({len(self.configs)} coins, {n_strats} strategies, regime + MTF).")

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

    # ── Multi-Timeframe Bias ──

    def _get_mtf_bias(self, symbol: str) -> dict:
        """
        Get multi-timeframe directional bias.

        - Daily (1d): Overall trend direction using EMA 21/50 crossover
        - Hourly (1h): Short-term momentum confirmation using RSI + EMA trend

        Returns:
            {
                "daily_bias": 1 (bullish) / -1 (bearish) / 0 (neutral),
                "hourly_bias": 1 / -1 / 0,
                "daily_trend": "bullish" / "bearish" / "neutral",
                "hourly_trend": "bullish" / "bearish" / "neutral",
                "confirmed": bool,  # True if daily and hourly agree
                "reason": str,
            }
        """
        # Check cache
        now = time.time()
        cached = self._mtf_cache.get(symbol)
        if cached and (now - cached["ts"]) < self.MTF_CACHE_TTL:
            return cached["data"]

        bias_labels = {1: "bullish", -1: "bearish", 0: "neutral"}
        default = {
            "daily_bias": 0, "hourly_bias": 0,
            "daily_trend": "neutral", "hourly_trend": "neutral",
            "confirmed": False, "reason": "MTF data unavailable",
        }

        # ── Daily bias: EMA 21 vs EMA 50 ──
        try:
            df_1d = self.fetch_latest(symbol, "1d", limit=100)
            if df_1d.empty or len(df_1d) < 55:
                logger.warning(f"  {symbol}: MTF daily data insufficient, skipping MTF filter")
                self._mtf_cache[symbol] = {"data": default, "ts": now}
                return default
            df_1d = add_all_indicators(df_1d)
            latest_1d = df_1d.iloc[-1]

            ema21_1d = latest_1d["ema_21"]
            ema50_1d = latest_1d["ema_50"]
            close_1d = latest_1d["close"]

            if ema21_1d > ema50_1d and close_1d > ema21_1d:
                daily_bias = 1
            elif ema21_1d < ema50_1d and close_1d < ema21_1d:
                daily_bias = -1
            else:
                daily_bias = 0
        except Exception as e:
            logger.warning(f"  {symbol}: MTF daily fetch failed ({e}), allowing signal through")
            self._mtf_cache[symbol] = {"data": default, "ts": now}
            return default

        # ── Hourly bias: RSI zone + EMA direction ──
        try:
            df_1h = self.fetch_latest(symbol, "1h", limit=100)
            if df_1h.empty or len(df_1h) < 55:
                logger.warning(f"  {symbol}: MTF hourly data insufficient")
                hourly_bias = 0
            else:
                df_1h = add_all_indicators(df_1h)
                latest_1h = df_1h.iloc[-1]

                rsi_1h = latest_1h["rsi"]
                ema21_1h = latest_1h["ema_21"]
                close_1h = latest_1h["close"]
                # EMA21 trending: compare current vs 3 bars ago
                ema21_3_ago = df_1h["ema_21"].iloc[-4] if len(df_1h) >= 4 else ema21_1h

                if rsi_1h > 45 and close_1h > ema21_1h and ema21_1h > ema21_3_ago:
                    hourly_bias = 1
                elif rsi_1h < 55 and close_1h < ema21_1h and ema21_1h < ema21_3_ago:
                    hourly_bias = -1
                else:
                    hourly_bias = 0
        except Exception as e:
            logger.warning(f"  {symbol}: MTF hourly fetch failed ({e})")
            hourly_bias = 0

        confirmed = (daily_bias == hourly_bias) and daily_bias != 0
        reason_parts = [
            f"1d={bias_labels[daily_bias]}",
            f"1h={bias_labels[hourly_bias]}",
        ]
        if confirmed:
            reason_parts.append("CONFIRMED")

        result = {
            "daily_bias": daily_bias,
            "hourly_bias": hourly_bias,
            "daily_trend": bias_labels[daily_bias],
            "hourly_trend": bias_labels[hourly_bias],
            "confirmed": confirmed,
            "reason": f"MTF: {', '.join(reason_parts)}",
        }

        self._mtf_cache[symbol] = {"data": result, "ts": now}
        logger.debug(f"  {symbol}: {result['reason']}")
        return result

    # ── Main Signal Logic ──

    def generate_signal(self, symbol: str) -> dict:
        """
        Generate signal for a symbol by checking all configured strategies.
        Priority: first strategy that fires wins (Momentum Reversal first).
        """
        if symbol not in self.configs:
            return {"signal": 0, "reason": "Not configured", "symbol": symbol}

        config = self.configs[symbol]
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
                # ── Multi-timeframe confirmation ──
                mtf = self._get_mtf_bias(symbol)

                # Hard filter: daily trend vetoes disagreeing signals
                if mtf["daily_bias"] != 0 and result["signal"] != mtf["daily_bias"]:
                    logger.info(
                        f"  {symbol}: Signal {result['signal']} rejected by daily trend "
                        f"({mtf['daily_trend']})"
                    )
                    continue  # Try next strategy

                # Soft filter: hourly warns but doesn't block
                if mtf["hourly_bias"] != 0 and result["signal"] != mtf["hourly_bias"]:
                    logger.info(
                        f"  {symbol}: Signal {result['signal']} not confirmed by 1h "
                        f"({mtf['hourly_trend']}), proceeding with caution"
                    )

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
                    "mtf_daily": mtf["daily_trend"],
                    "mtf_hourly": mtf["hourly_trend"],
                    "mtf_confirmed": mtf["confirmed"],
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

        # No signal from MR/VB/MTF — try Funding Rate strategy
        try:
            fr_result = self.funding_rate_strategy.check_signal(symbol, df)
            if fr_result["signal"] != 0 and fr_result.get("confidence", 0) >= 0.6:
                # MTF confirmation (soft — FR is already contrarian)
                mtf = self._get_mtf_bias(symbol)
                return {
                    **fr_result,
                    "symbol": symbol,
                    "timeframe": config["timeframe"],
                    "stop_loss": 0.025,  # Tighter stop for FR trades
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
                    "mtf_daily": mtf.get("daily_trend", "neutral"),
                    "mtf_hourly": mtf.get("hourly_trend", "neutral"),
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    ),
                }
        except Exception as e:
            logger.debug(f"  {symbol}: Funding rate check failed: {e}")

        # Try Liquidation Cascade strategy
        try:
            lc_result = self.liquidation_strategy.check_signal(symbol, df)
            if lc_result["signal"] != 0 and lc_result.get("confidence", 0) >= 0.6:
                return {
                    **lc_result,
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
        except Exception as e:
            logger.debug(f"  {symbol}: Liquidation cascade check failed: {e}")

        # Try ML Signal model (lowest priority — needs trained model)
        try:
            ml_result = self.ml_model.check_signal(symbol, df)
            if ml_result["signal"] != 0 and ml_result.get("confidence", 0) >= 0.45:
                # ML also respects MTF daily trend
                mtf = self._get_mtf_bias(symbol)
                if mtf["daily_bias"] != 0 and ml_result["signal"] != mtf["daily_bias"]:
                    logger.info(
                        f"  {symbol}: ML signal {ml_result['signal']} rejected by daily trend"
                    )
                else:
                    return {
                        **ml_result,
                        "symbol": symbol,
                        "timeframe": config["timeframe"],
                        "stop_loss": 0.025,
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
                        "mtf_daily": mtf.get("daily_trend", "neutral"),
                        "mtf_hourly": mtf.get("hourly_trend", "neutral"),
                        "timestamp": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S UTC"
                        ),
                    }
        except Exception as e:
            logger.debug(f"  {symbol}: ML signal check failed: {e}")

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
        for symbol in self.configs:
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
    print(f"  CURRENT SIGNALS (v3 — {len(ALPHA_CONFIGS)} coins, 3 strategies)")
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