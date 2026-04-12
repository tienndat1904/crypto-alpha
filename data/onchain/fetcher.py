"""
On-Chain Data Fetcher — Free APIs
====================================
Fetches on-chain metrics from free sources:
  1. Alternative.me — Fear & Greed Index (daily, full history)
  2. Blockchain.com — BTC hash rate, difficulty, mempool (daily)
  3. CoinGecko — Market dominance, total market cap, volume (daily)

These metrics serve as FILTERS for existing strategies,
not standalone alpha signals (daily resolution, 24h delay).

Usage:
    from data.onchain.fetcher import OnchainFetcher
    fetcher = OnchainFetcher()
    fng = fetcher.fetch_fear_greed(days=730)
    market = fetcher.fetch_market_data()
"""

import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import requests
from loguru import logger


class OnchainFetcher:
    """Fetches on-chain and sentiment data from free APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "crypto-alpha-bot/1.0",
            "Accept": "application/json",
        })
        logger.info("OnchainFetcher initialized.")

    # ═══════════════════════════════════════════
    # 1. FEAR & GREED INDEX (Alternative.me)
    # ═══════════════════════════════════════════

    def fetch_fear_greed(self, days: int = 730) -> pd.DataFrame:
        """
        Fetch Crypto Fear & Greed Index history.
        
        Values: 0 = Extreme Fear, 100 = Extreme Greed
        - Extreme Fear (<25): potential buy signal
        - Extreme Greed (>75): potential sell signal

        Returns:
            DataFrame: timestamp, fng_value, fng_class
        """
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()["data"]

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
            df["fng_value"] = df["value"].astype(int)
            df["fng_class"] = df["value_classification"]
            df = df[["timestamp", "fng_value", "fng_class"]]
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Fear & Greed: fetched {len(df)} days ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
            return df

        except Exception as e:
            logger.error(f"Fear & Greed fetch failed: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════
    # 2. COINGECKO — Market Overview
    # ═══════════════════════════════════════════

    def fetch_btc_dominance(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch BTC dominance and total market cap history from CoinGecko.
        
        BTC dominance rising = flight to safety (fear)
        BTC dominance falling = alt season (greed/risk-on)

        Returns:
            DataFrame: timestamp, total_mcap, btc_mcap, btc_dominance
        """
        try:
            # Total market cap
            url_total = "https://api.coingecko.com/api/v3/global/market_cap_chart"
            params = {"days": days, "vs_currency": "usd"}

            # CoinGecko free: /coins/bitcoin/market_chart for BTC
            url_btc = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params_btc = {"vs_currency": "usd", "days": str(days)}

            resp_btc = self.session.get(url_btc, params=params_btc, timeout=30)
            resp_btc.raise_for_status()
            btc_data = resp_btc.json()

            # Extract market cap
            mcap = btc_data.get("market_caps", [])
            df = pd.DataFrame(mcap, columns=["timestamp_ms", "btc_mcap"])
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df[["timestamp", "btc_mcap"]]

            # Extract volume
            volumes = btc_data.get("total_volumes", [])
            vol_df = pd.DataFrame(volumes, columns=["timestamp_ms", "btc_volume_usd"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp_ms"], unit="ms", utc=True)
            df = df.merge(vol_df[["timestamp", "btc_volume_usd"]], on="timestamp", how="left")

            # Extract price
            prices = btc_data.get("prices", [])
            price_df = pd.DataFrame(prices, columns=["timestamp_ms", "btc_price"])
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp_ms"], unit="ms", utc=True)
            df = df.merge(price_df[["timestamp", "btc_price"]], on="timestamp", how="left")

            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"CoinGecko BTC: fetched {len(df)} days")

            time.sleep(1.5)  # Rate limit: 10-30 req/min on free tier
            return df

        except Exception as e:
            logger.error(f"CoinGecko fetch failed: {e}")
            return pd.DataFrame()

    def fetch_eth_data(self, days: int = 365) -> pd.DataFrame:
        """Fetch ETH market chart from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            params = {"vs_currency": "usd", "days": days, "interval": "daily"}

            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            prices = pd.DataFrame(data["prices"], columns=["timestamp_ms", "eth_price"])
            volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp_ms", "eth_volume_usd"])

            prices["timestamp"] = pd.to_datetime(prices["timestamp_ms"], unit="ms", utc=True)
            volumes["timestamp"] = pd.to_datetime(volumes["timestamp_ms"], unit="ms", utc=True)

            df = prices[["timestamp", "eth_price"]].merge(
                volumes[["timestamp", "eth_volume_usd"]], on="timestamp", how="left"
            )

            # ETH/BTC ratio (useful for alt season detection)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"CoinGecko ETH: fetched {len(df)} days")

            time.sleep(1.5)
            return df

        except Exception as e:
            logger.error(f"CoinGecko ETH fetch failed: {e}")
            return pd.DataFrame()

    # ═══════════════════════════════════════════
    # 3. DERIVED METRICS
    # ═══════════════════════════════════════════

    def compute_onchain_features(
        self,
        fng_df: pd.DataFrame,
        btc_df: pd.DataFrame,
        eth_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute derived on-chain features from raw data.

        Features:
            - fng_value: raw Fear & Greed (0-100)
            - fng_sma_7: 7-day moving average of F&G
            - fng_extreme_fear: binary, F&G < 25
            - fng_extreme_greed: binary, F&G > 75
            - fng_momentum: 7-day change in F&G
            - btc_volume_change: % change in BTC volume
            - eth_btc_ratio: ETH/BTC price ratio (alt season indicator)
            - eth_btc_ratio_change: 7-day change in ratio
        """
        # Start with Fear & Greed
        df = fng_df.copy()
        df["date"] = df["timestamp"].dt.date

        # F&G features
        df["fng_sma_7"] = df["fng_value"].rolling(7).mean()
        df["fng_sma_30"] = df["fng_value"].rolling(30).mean()
        df["fng_extreme_fear"] = (df["fng_value"] < 25).astype(int)
        df["fng_extreme_greed"] = (df["fng_value"] > 75).astype(int)
        df["fng_momentum"] = df["fng_value"].diff(7)
        df["fng_zscore"] = (
            (df["fng_value"] - df["fng_value"].rolling(30).mean())
            / df["fng_value"].rolling(30).std()
        )

        # Merge BTC data
        if not btc_df.empty:
            btc_df = btc_df.copy()
            btc_df["date"] = btc_df["timestamp"].dt.date
            btc_features = btc_df[["date", "btc_mcap", "btc_volume_usd", "btc_price"]].copy()
            btc_features["btc_volume_change"] = btc_features["btc_volume_usd"].pct_change(7)
            btc_features["btc_mcap_change"] = btc_features["btc_mcap"].pct_change(7)
            df = df.merge(btc_features, on="date", how="left")

        # Merge ETH data for ETH/BTC ratio
        if eth_df is not None and not eth_df.empty:
            eth_df = eth_df.copy()
            eth_df["date"] = eth_df["timestamp"].dt.date
            eth_features = eth_df[["date", "eth_price", "eth_volume_usd"]].copy()
            df = df.merge(eth_features, on="date", how="left")

            if "btc_price" in df.columns:
                df["eth_btc_ratio"] = df["eth_price"] / df["btc_price"]
                df["eth_btc_ratio_sma"] = df["eth_btc_ratio"].rolling(14).mean()
                df["eth_btc_ratio_change"] = df["eth_btc_ratio"].pct_change(7)

        df = df.set_index("timestamp")
        logger.info(f"Computed {len(df.columns)} on-chain features over {len(df)} days")

        return df

    # ═══════════════════════════════════════════
    # MASTER FETCH
    # ═══════════════════════════════════════════

    def fetch_all(self, days: int = 730) -> pd.DataFrame:
        """
        Fetch all on-chain data and compute features.

        Returns:
            DataFrame with all on-chain features, daily resolution.
        """
        logger.info(f"Fetching all on-chain data ({days} days)...")

        fng = self.fetch_fear_greed(days)
        btc = self.fetch_btc_dominance(days)
        eth = self.fetch_eth_data(days)

        if fng.empty:
            logger.error("Fear & Greed data is required. Aborting.")
            return pd.DataFrame()

        features = self.compute_onchain_features(fng, btc, eth)
        return features


if __name__ == "__main__":
    fetcher = OnchainFetcher()

    print("\n=== Fetching On-Chain Data ===\n")
    df = fetcher.fetch_all(days=365)

    if not df.empty:
        print(f"\nTotal features: {len(df.columns)}")
        print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"\nLatest values:")
        latest = df.iloc[-1]
        print(f"  Fear & Greed:     {latest.get('fng_value', 'N/A')} ({latest.get('fng_class', 'N/A')})")
        print(f"  F&G SMA(7):       {latest.get('fng_sma_7', 'N/A'):.1f}")
        print(f"  Extreme Fear:     {'YES' if latest.get('fng_extreme_fear', 0) else 'NO'}")
        print(f"  Extreme Greed:    {'YES' if latest.get('fng_extreme_greed', 0) else 'NO'}")
        if "btc_price" in df.columns:
            print(f"  BTC Price:        ${latest['btc_price']:,.0f}")
        if "eth_btc_ratio" in df.columns:
            print(f"  ETH/BTC Ratio:    {latest['eth_btc_ratio']:.4f}")
    print()
