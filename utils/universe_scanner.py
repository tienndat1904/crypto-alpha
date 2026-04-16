"""
Universe Scanner
================
Scans Binance for top tradeable coins by volume and volatility.
Runs weekly to suggest universe updates.

Usage:
    from utils.universe_scanner import UniverseScanner
    scanner = UniverseScanner()
    rankings = scanner.scan()
    scanner.suggest_updates()
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from loguru import logger


# Coins to always keep (blue chips, high liquidity)
CORE_COINS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"}

# Coins to never include (stablecoins, wrapped, low-cap risk)
BLACKLIST = {
    "USDC/USDT", "BUSD/USDT", "TUSD/USDT", "FDUSD/USDT",
    "WBTC/USDT", "WETH/USDT", "STETH/USDT",
}

# Minimum requirements
MIN_24H_VOLUME_USD = 50_000_000  # $50M daily volume
MIN_MARKET_CAP_RANK = 100        # Top 100 by market cap (if available)
MAX_UNIVERSE_SIZE = 15
MIN_UNIVERSE_SIZE = 8


class UniverseScanner:
    """Scans Binance for top tradeable coins and ranks them by a composite score."""

    def __init__(self, min_volume: float = MIN_24H_VOLUME_USD,
                 max_universe: int = MAX_UNIVERSE_SIZE):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.min_volume = min_volume
        self.max_universe = max_universe

    def _fetch_tickers(self) -> dict:
        """Fetch all tickers from Binance."""
        logger.info("Fetching all tickers from Binance...")
        tickers = self.exchange.fetch_tickers()
        return tickers

    def _filter_usdt_pairs(self, tickers: dict) -> list[dict]:
        """Filter to USDT pairs that meet volume and blacklist criteria."""
        candidates = []
        for symbol, ticker in tickers.items():
            # Only USDT pairs
            if not symbol.endswith("/USDT"):
                continue
            # Skip blacklisted
            if symbol in BLACKLIST:
                continue
            # Skip leveraged tokens (e.g. BTCUP, BTCDOWN)
            base = symbol.split("/")[0]
            if base.endswith(("UP", "DOWN", "BEAR", "BULL")):
                continue
            # Skip non-ASCII symbols (e.g. 币安人生) and very short/long bases
            if not base.isascii() or len(base) < 2 or len(base) > 10:
                continue
            # Skip stablecoin-like tokens
            if base in ("USD1", "USDE", "USDP", "DAI", "FRAX", "LUSD", "PYUSD",
                       "RLUSD", "XAUT", "PAXG"):
                continue

            quote_vol = ticker.get("quoteVolume") or 0
            if quote_vol < self.min_volume:
                continue

            candidates.append({
                "symbol": symbol,
                "volume_24h": quote_vol,
                "last_price": ticker.get("last") or 0,
                "bid": ticker.get("bid") or 0,
                "ask": ticker.get("ask") or 0,
            })

        # Sort by volume descending
        candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
        logger.info(f"Found {len(candidates)} USDT pairs above ${self.min_volume/1e6:.0f}M volume")
        return candidates

    def _fetch_candles(self, symbol: str, timeframe: str = "4h",
                       limit: int = 42) -> pd.DataFrame:
        """
        Fetch recent candles for a symbol.
        Default: 42 x 4h candles = 7 days of data.
        """
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                candles,
                columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.drop(columns=["timestamp_ms"]).set_index("timestamp")
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {symbol}: {e}")
            return pd.DataFrame()

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """
        Compute ranking metrics from a 7-day candle DataFrame.

        Returns dict with:
          - volatility_7d: std of log returns (annualized-ish for ranking)
          - momentum_7d: 7-day rate of change (%)
          - atr_pct: average true range as % of close
        """
        if df.empty or len(df) < 10:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Log returns
        log_returns = np.log(close / close.shift(1)).dropna()
        volatility = log_returns.std()

        # 7-day momentum: last close vs first close
        momentum = (close.iloc[-1] / close.iloc[0] - 1) * 100

        # ATR%
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean().iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100

        return {
            "volatility_7d": volatility,
            "momentum_7d": momentum,
            "atr_pct": atr_pct,
        }

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Min-max normalize a Series to [0, 1]."""
        smin = series.min()
        smax = series.max()
        if smax == smin:
            return pd.Series(0.5, index=series.index)
        return (series - smin) / (smax - smin)

    def scan(self, top_n_by_volume: int = 50) -> pd.DataFrame:
        """
        Scan all USDT pairs on Binance and rank by composite score.

        Score = weighted combination of:
          - 24h volume (normalized)       -- weight 0.3
          - 7d volatility (ATR%)          -- weight 0.3
          - 7d price momentum (ROC)       -- weight 0.2
          - Spread tightness              -- weight 0.2

        Returns DataFrame sorted by score descending.
        """
        tickers = self._fetch_tickers()
        candidates = self._filter_usdt_pairs(tickers)

        # Take top N by volume for detailed analysis
        top_candidates = candidates[:top_n_by_volume]

        rows = []
        for i, cand in enumerate(top_candidates):
            symbol = cand["symbol"]
            logger.info(f"  [{i+1}/{len(top_candidates)}] Analyzing {symbol}...")

            df = self._fetch_candles(symbol)
            metrics = self._compute_metrics(df)
            if metrics is None:
                logger.debug(f"    Skipped {symbol}: insufficient candle data")
                continue

            # Spread calculation
            bid = cand["bid"]
            ask = cand["ask"]
            mid = (bid + ask) / 2 if (bid + ask) > 0 else cand["last_price"]
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999

            rows.append({
                "symbol": symbol,
                "volume_24h": cand["volume_24h"],
                "volatility_7d": metrics["volatility_7d"],
                "momentum_7d": metrics["momentum_7d"],
                "atr_pct": metrics["atr_pct"],
                "spread_pct": spread_pct,
            })

        if not rows:
            logger.warning("No coins passed the scan filters")
            return pd.DataFrame()

        df_rank = pd.DataFrame(rows)

        # Normalize each metric to [0, 1]
        vol_norm = self._normalize(df_rank["volume_24h"])
        volatility_norm = self._normalize(df_rank["volatility_7d"])
        momentum_norm = self._normalize(df_rank["momentum_7d"])
        # Spread: lower is better, so invert
        spread_norm = 1 - self._normalize(df_rank["spread_pct"])

        # Composite score
        df_rank["score"] = (
            0.3 * vol_norm +
            0.3 * volatility_norm +
            0.2 * momentum_norm +
            0.2 * spread_norm
        )

        df_rank = df_rank.sort_values("score", ascending=False).reset_index(drop=True)
        df_rank.index = df_rank.index + 1  # 1-based rank
        df_rank.index.name = "rank"

        logger.info(f"Scan complete: {len(df_rank)} coins ranked")
        return df_rank

    def get_current_universe(self) -> list:
        """Get the current trading universe from ALPHA_CONFIGS."""
        from trading.signal_generator import ALPHA_CONFIGS
        return list(ALPHA_CONFIGS.keys())

    def suggest_updates(self) -> dict:
        """
        Compare current universe with scan results.

        Returns:
            {
                "keep": [...],       # Current coins still ranked well
                "add": [...],        # New coins to consider adding
                "remove": [...],     # Current coins that dropped in ranking
                "rankings": DataFrame,
                "scan_date": str,
            }
        """
        rankings = self.scan()
        if rankings.empty:
            return {
                "keep": [], "add": [], "remove": [],
                "rankings": rankings,
                "scan_date": datetime.now(timezone.utc).isoformat(),
            }

        current = set(self.get_current_universe())
        top_symbols = set(rankings.head(self.max_universe).symbol)
        top_30_symbols = set(rankings.head(30).symbol)

        keep = []
        remove = []
        add = []

        for coin in current:
            if coin in CORE_COINS:
                keep.append(coin)
            elif coin in top_30_symbols:
                keep.append(coin)
            else:
                remove.append(coin)

        # Always keep core coins even if not in current universe
        for coin in CORE_COINS:
            if coin not in keep:
                keep.append(coin)

        # Suggest new coins from top rankings not already in universe
        slots_available = self.max_universe - len(keep)
        for _, row in rankings.iterrows():
            if slots_available <= 0:
                break
            sym = row["symbol"]
            if sym not in current and sym not in CORE_COINS and sym not in keep:
                add.append(sym)
                slots_available -= 1

        result = {
            "keep": sorted(keep),
            "add": add,
            "remove": sorted(remove),
            "rankings": rankings,
            "scan_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

        logger.info(f"Universe suggestions: keep={len(keep)}, add={len(add)}, remove={len(remove)}")
        return result

    def generate_config(self, symbol: str) -> dict:
        """
        Generate an ALPHA_CONFIGS entry for a new coin based on its characteristics.

        Uses the coin's volatility to determine appropriate parameters:
        - High volatility: wider stop (4%), higher roc_threshold (-10)
        - Low volatility: tighter stop (3%), lower roc_threshold (-8)
        """
        df = self._fetch_candles(symbol, timeframe="4h", limit=42)
        metrics = self._compute_metrics(df)

        if metrics is None:
            logger.warning(f"Cannot generate config for {symbol}: no data")
            return {}

        vol = metrics["volatility_7d"]

        # Determine volatility regime:
        # High vol threshold: std of 4h log returns > 0.03 (~4.2% daily)
        high_vol = vol > 0.03

        if high_vol:
            roc_threshold = -10.0
            stop_loss_mr = 0.03
            stop_loss_vb = 0.04
            atr_multiplier = 1.5
        else:
            roc_threshold = -8.0
            stop_loss_mr = 0.03
            stop_loss_vb = 0.04
            atr_multiplier = 2.0

        config = {
            "strategies": [
                {
                    "name": "momentum_reversal",
                    "params": {
                        "roc_threshold": roc_threshold,
                        "roc_exit": 3.0,
                        "support_buffer": 0.05,
                    },
                    "stop_loss": stop_loss_mr,
                },
                {
                    "name": "volatility_breakout",
                    "params": {
                        "atr_multiplier": atr_multiplier,
                        "volume_threshold": 1.5,
                        "holding_periods": 6,
                    },
                    "stop_loss": stop_loss_vb,
                },
            ],
            "timeframe": "4h",
            "lookback_candles": 250,
        }

        logger.info(
            f"Generated config for {symbol}: vol={'HIGH' if high_vol else 'LOW'} "
            f"(std={vol:.4f}), roc_thresh={roc_threshold}, atr_mult={atr_multiplier}"
        )
        return config


if __name__ == "__main__":
    import sys

    scanner = UniverseScanner()

    print("\n" + "=" * 80)
    print("  UNIVERSE SCANNER - Binance USDT Pairs")
    print("=" * 80)

    # Run scan
    rankings = scanner.scan()

    if rankings.empty:
        print("\n  No coins found. Check network connection / Binance availability.")
        sys.exit(1)

    # Print rankings table
    print(f"\n  Top {len(rankings)} coins by composite score:\n")
    print(f"  {'Rank':<6} {'Symbol':<12} {'Volume 24h':>14} {'Vol 7d':>8} "
          f"{'Mom 7d':>8} {'ATR%':>7} {'Spread%':>9} {'Score':>7}")
    print("  " + "-" * 73)

    for rank, row in rankings.iterrows():
        vol_m = row["volume_24h"] / 1e6
        is_core = " *" if row["symbol"] in CORE_COINS else ""
        print(
            f"  {rank:<6} {row['symbol']:<12} ${vol_m:>11,.1f}M "
            f"{row['volatility_7d']:>7.4f} {row['momentum_7d']:>+7.1f}% "
            f"{row['atr_pct']:>6.2f}% {row['spread_pct']:>8.4f}% "
            f"{row['score']:>6.3f}{is_core}"
        )

    print(f"\n  * = Core coin (always kept)")

    # Suggest updates
    print("\n" + "=" * 80)
    print("  UNIVERSE UPDATE SUGGESTIONS")
    print("=" * 80)

    suggestions = scanner.suggest_updates()

    print(f"\n  Scan date: {suggestions['scan_date']}")
    print(f"\n  KEEP ({len(suggestions['keep'])}): {', '.join(suggestions['keep'])}")
    print(f"  ADD  ({len(suggestions['add'])}): {', '.join(suggestions['add']) or '(none)'}")
    print(f"  DROP ({len(suggestions['remove'])}): {', '.join(suggestions['remove']) or '(none)'}")

    # Generate configs for suggested additions
    if suggestions["add"]:
        print("\n" + "-" * 80)
        print("  SUGGESTED CONFIGS FOR NEW COINS")
        print("-" * 80)
        for sym in suggestions["add"][:3]:  # Show top 3
            cfg = scanner.generate_config(sym)
            if cfg:
                mr = cfg["strategies"][0]
                vb = cfg["strategies"][1]
                print(f"\n  {sym}:")
                print(f"    MR: roc_thresh={mr['params']['roc_threshold']}, stop={mr['stop_loss']}")
                print(f"    VB: atr_mult={vb['params']['atr_multiplier']}, stop={vb['stop_loss']}")

    print()
