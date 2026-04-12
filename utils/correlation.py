"""
Correlation Filter
===================
Prevents opening highly correlated positions simultaneously.
Crypto assets are heavily correlated — during crashes, all positions
move together, amplifying drawdown beyond what single-asset backtests show.

Usage:
    from utils.correlation import CorrelationFilter
    cf = CorrelationFilter()
    allowed, reason = cf.can_open_position("SOL/USDT", existing=["BTC/USDT", "ETH/USDT"])
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from loguru import logger

from config.settings import (
    CORRELATION_THRESHOLD,
    CORRELATION_LOOKBACK_DAYS,
)
from data.models import engine


class CorrelationFilter:
    """
    Checks pairwise return correlation between a candidate symbol
    and already-open positions. Blocks entry if correlation exceeds threshold.
    """

    def __init__(
        self,
        threshold: float = CORRELATION_THRESHOLD,
        lookback_days: int = CORRELATION_LOOKBACK_DAYS,
        timeframe: str = "4h",
    ):
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.timeframe = timeframe
        self._cache = {}

    def _load_returns(self, symbol: str) -> pd.Series:
        """Load recent close prices and compute returns."""
        if symbol in self._cache:
            return self._cache[symbol]

        query = text(
            "SELECT timestamp, close FROM ohlcv "
            "WHERE symbol = :symbol AND timeframe = :tf "
            "ORDER BY timestamp DESC LIMIT :limit"
        )

        # 4h = 6 candles/day, 1d = 1 candle/day
        candles_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(self.timeframe, 6)
        limit = self.lookback_days * candles_per_day

        try:
            with engine.connect() as conn:
                df = pd.read_sql(
                    query, conn,
                    params={"symbol": symbol, "tf": self.timeframe, "limit": limit},
                )
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return None

        if len(df) < 20:
            logger.warning(f"Not enough data for {symbol}: {len(df)} rows")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        returns = df["close"].pct_change().dropna()
        returns.name = symbol

        self._cache[symbol] = returns
        return returns

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Compute pairwise Pearson correlation of returns."""
        ret_a = self._load_returns(symbol_a)
        ret_b = self._load_returns(symbol_b)

        if ret_a is None or ret_b is None:
            return 0.0

        # Align on common timestamps
        combined = pd.concat([ret_a, ret_b], axis=1, join="inner")
        if len(combined) < 20:
            return 0.0

        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        return corr if not np.isnan(corr) else 0.0

    def get_correlation_matrix(self, symbols: list) -> pd.DataFrame:
        """Compute full correlation matrix for a list of symbols."""
        returns_dict = {}
        for sym in symbols:
            ret = self._load_returns(sym)
            if ret is not None:
                returns_dict[sym] = ret

        if len(returns_dict) < 2:
            return pd.DataFrame()

        combined = pd.DataFrame(returns_dict).dropna()
        return combined.corr()

    def can_open_position(
        self,
        candidate: str,
        open_positions: list,
    ) -> tuple:
        """
        Check if opening a position on candidate is allowed.

        Args:
            candidate: Symbol to check (e.g. "SOL/USDT")
            open_positions: List of symbols currently held

        Returns:
            (allowed: bool, reason: str)
        """
        if not open_positions:
            return True, "No existing positions"

        high_corr_pairs = []

        for existing in open_positions:
            if existing == candidate:
                return False, f"Already have position in {candidate}"

            corr = self.get_correlation(candidate, existing)
            logger.debug(f"Correlation {candidate} vs {existing}: {corr:.3f}")

            if abs(corr) > self.threshold:
                high_corr_pairs.append((existing, corr))

        if high_corr_pairs:
            details = ", ".join(
                [f"{sym} (r={c:.2f})" for sym, c in high_corr_pairs]
            )
            return False, (
                f"Blocked: {candidate} too correlated with {details}. "
                f"Threshold={self.threshold}"
            )

        return True, "OK"

    def clear_cache(self):
        """Clear cached returns data."""
        self._cache.clear()
