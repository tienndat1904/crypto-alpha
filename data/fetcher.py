"""
Data Fetcher - Binance OHLCV Pipeline
======================================
Fetches historical and incremental candlestick data from Binance
via CCXT and stores it in MySQL.

Usage:
    # First run - fetch 2 years of history for all coins
    python -m data.fetcher --full

    # Incremental update - fetch only new candles
    python -m data.fetcher --update

    # Fetch specific symbol
    python -m data.fetcher --full --symbol BTC/USDT
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from loguru import logger
from sqlalchemy import text
from tqdm import tqdm

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    COIN_UNIVERSE,
    TIMEFRAMES,
    HISTORY_DAYS,
    FETCH_BATCH_SIZE,
    LOG_FILE,
)
from data.models import OHLCV, SessionLocal, init_db

# ── Configure Logging ──
logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
)


class BinanceFetcher:
    """
    Fetches OHLCV data from Binance and stores in MySQL.
    
    Features:
    - Pagination: handles Binance's 1000-candle limit per request
    - Incremental updates: only fetches new candles since last stored
    - Rate limiting: respects Binance API limits
    - Duplicate handling: uses INSERT IGNORE for idempotent writes
    """

    def __init__(self):
        """Initialize Binance exchange connection via CCXT."""
        self.exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY if BINANCE_API_KEY else None,
            "secret": BINANCE_SECRET_KEY if BINANCE_SECRET_KEY else None,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        logger.info("Binance exchange connection initialized.")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol/timeframe with automatic pagination.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            timeframe: Candle interval, e.g. "1h", "4h", "1d"
            since: Start datetime (UTC)
            until: End datetime (UTC), defaults to now

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if until is None:
            until = datetime.now(timezone.utc)

        since_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)

        all_candles = []
        current_since = since_ms

        while current_since < until_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=FETCH_BATCH_SIZE,
                )
            except ccxt.RateLimitExceeded:
                logger.warning(f"Rate limit hit for {symbol}. Waiting 60s...")
                time.sleep(60)
                continue
            except ccxt.NetworkError as e:
                logger.error(f"Network error for {symbol}: {e}. Retrying in 10s...")
                time.sleep(10)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error for {symbol}: {e}. Skipping.")
                break

            if not candles:
                break

            all_candles.extend(candles)

            # Move cursor to after the last candle
            last_ts = candles[-1][0]
            if last_ts == current_since:
                # No progress, avoid infinite loop
                break
            current_since = last_ts + 1

            # Small delay to be polite to the API
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )

        # Convert millisecond timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp_ms"])

        # Remove candles beyond 'until'
        df = df[df["timestamp"] <= until]

        # Drop duplicates (can happen at pagination boundaries)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_last_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """
        Get the most recent candle timestamp stored in DB for a symbol/timeframe.

        Returns:
            datetime if data exists, None otherwise
        """
        session = SessionLocal()
        try:
            result = session.query(OHLCV.timestamp).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe,
            ).order_by(OHLCV.timestamp.desc()).first()

            if result:
                ts = result[0]
                # Ensure timezone-aware
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            return None
        finally:
            session.close()

    def save_to_db(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Save OHLCV DataFrame to MySQL. Skips duplicates automatically.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair
            timeframe: Candle interval

        Returns:
            Number of new rows inserted
        """
        if df.empty:
            return 0

        session = SessionLocal()
        inserted = 0

        try:
            for _, row in df.iterrows():
                ts = row["timestamp"].to_pydatetime().replace(tzinfo=None)

                # Check if candle already exists
                exists = session.query(OHLCV.id).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timeframe == timeframe,
                    OHLCV.timestamp == ts,
                ).first()

                if exists:
                    continue

                candle = OHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                session.add(candle)
                inserted += 1

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"DB error saving {symbol} {timeframe}: {e}")
            raise
        finally:
            session.close()

        return inserted

    def fetch_full_history(self, symbols: list = None, timeframes: list = None):
        """
        Fetch full historical data (2 years) for all symbols and timeframes.

        Args:
            symbols: List of trading pairs (default: COIN_UNIVERSE)
            timeframes: List of timeframes (default: TIMEFRAMES)
        """
        symbols = symbols or COIN_UNIVERSE
        timeframes = timeframes or TIMEFRAMES

        since = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)

        total_tasks = len(symbols) * len(timeframes)
        logger.info(
            f"Starting FULL fetch: {len(symbols)} symbols × "
            f"{len(timeframes)} timeframes = {total_tasks} tasks"
        )
        logger.info(f"History from: {since.strftime('%Y-%m-%d')}")

        with tqdm(total=total_tasks, desc="Fetching data") as pbar:
            for symbol in symbols:
                for tf in timeframes:
                    pbar.set_postfix_str(f"{symbol} {tf}")

                    try:
                        df = self.fetch_ohlcv(symbol, tf, since)
                        if df.empty:
                            logger.warning(f"No data for {symbol} {tf}")
                            pbar.update(1)
                            continue

                        inserted = self.save_to_db(df, symbol, tf)
                        logger.info(
                            f"{symbol} {tf}: fetched {len(df)} candles, "
                            f"inserted {inserted} new"
                        )
                    except Exception as e:
                        logger.error(f"Failed {symbol} {tf}: {e}")

                    pbar.update(1)

        self.print_summary()

    def fetch_update(self, symbols: list = None, timeframes: list = None):
        """
        Incremental update: fetch only new candles since last stored.

        Args:
            symbols: List of trading pairs (default: COIN_UNIVERSE)
            timeframes: List of timeframes (default: TIMEFRAMES)
        """
        symbols = symbols or COIN_UNIVERSE
        timeframes = timeframes or TIMEFRAMES

        total_tasks = len(symbols) * len(timeframes)
        logger.info(f"Starting INCREMENTAL update: {total_tasks} tasks")

        total_inserted = 0

        with tqdm(total=total_tasks, desc="Updating data") as pbar:
            for symbol in symbols:
                for tf in timeframes:
                    pbar.set_postfix_str(f"{symbol} {tf}")

                    try:
                        last_ts = self.get_last_timestamp(symbol, tf)
                        if last_ts is None:
                            # No data yet, fetch full history
                            since = datetime.now(timezone.utc) - timedelta(
                                days=HISTORY_DAYS
                            )
                            logger.info(
                                f"No data for {symbol} {tf}, "
                                f"fetching full history"
                            )
                        else:
                            since = last_ts

                        df = self.fetch_ohlcv(symbol, tf, since)
                        if df.empty:
                            pbar.update(1)
                            continue

                        inserted = self.save_to_db(df, symbol, tf)
                        total_inserted += inserted

                        if inserted > 0:
                            logger.info(
                                f"{symbol} {tf}: +{inserted} new candles"
                            )
                    except Exception as e:
                        logger.error(f"Update failed {symbol} {tf}: {e}")

                    pbar.update(1)

        logger.info(f"Update complete. Total new candles: {total_inserted}")
        self.print_summary()

    def print_summary(self):
        """Print a summary of data stored in the database."""
        session = SessionLocal()
        try:
            result = session.execute(text("""
                SELECT symbol, timeframe, 
                       COUNT(*) as candles,
                       MIN(timestamp) as first_candle,
                       MAX(timestamp) as last_candle
                FROM ohlcv
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """))

            rows = result.fetchall()
            if not rows:
                logger.info("Database is empty.")
                return

            print("\n" + "=" * 75)
            print(f"{'Symbol':<12} {'TF':<5} {'Candles':>8}  "
                  f"{'From':<12} {'To':<12}")
            print("-" * 75)

            for row in rows:
                symbol, tf, count, first, last = row
                print(
                    f"{symbol:<12} {tf:<5} {count:>8}  "
                    f"{str(first)[:10]:<12} {str(last)[:10]:<12}"
                )

            # Total
            total = session.execute(
                text("SELECT COUNT(*) FROM ohlcv")
            ).scalar()
            print("-" * 75)
            print(f"{'TOTAL':<18} {total:>8}")
            print("=" * 75 + "\n")

        finally:
            session.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Alpha - Binance Data Fetcher"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fetch full 2-year history for all coins",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incremental update (only new candles)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Fetch specific symbol only, e.g. BTC/USDT",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print database summary",
    )

    args = parser.parse_args()

    # Initialize database
    init_db()

    fetcher = BinanceFetcher()

    symbols = [args.symbol] if args.symbol else None

    if args.full:
        fetcher.fetch_full_history(symbols=symbols)
    elif args.update:
        fetcher.fetch_update(symbols=symbols)
    elif args.summary:
        fetcher.print_summary()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m data.fetcher --full              # Fetch all history")
        print("  python -m data.fetcher --update            # Update new candles")
        print("  python -m data.fetcher --full --symbol BTC/USDT  # Single coin")
        print("  python -m data.fetcher --summary           # Show DB stats")


if __name__ == "__main__":
    main()
