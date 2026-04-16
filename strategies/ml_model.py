"""
ML Signal Model
================
LightGBM classifier trained on historical OHLCV + indicators to predict
BUY/SELL/HOLD signals.

Training pipeline:
  1. Fetch historical data with all indicators
  2. Label: future 6-candle return > +2% = BUY(1), < -2% = SHORT(-1), else HOLD(0)
  3. Features: RSI, ROC, ATR%, volume_ratio, price_position, bb_pct_b, ADX, etc.
  4. Train LightGBM with walk-forward validation
  5. Save model to models/ directory

Prediction:
  - Load trained model
  - Feed current candle features → predict signal + probability

Usage:
    # Train
    python -m strategies.ml_model --train

    # Use in signal generator
    from strategies.ml_model import MLSignalModel
    model = MLSignalModel()
    result = model.check_signal(symbol, df)
"""

import os
import sys
sys.path.insert(0, ".")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not installed. Run: pip install lightgbm scikit-learn joblib")

from config.settings import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Features used for training/prediction
FEATURE_COLS = [
    # Momentum
    "rsi", "roc_5", "roc_10", "roc_20",
    "stoch_k", "stoch_d",
    # Trend
    "adx", "plus_di", "minus_di",
    "macd", "macd_signal", "macd_histogram",
    # Volatility
    "atr_pct", "bb_width", "bb_pct_b",
    "volatility_14",
    # Volume
    "volume_ratio",
    # Price action
    "price_position", "candle_body_ratio",
    "returns", "log_returns",
    # Statistical
    "zscore",
]

# Labels
LABEL_MAP = {1: "BUY", 0: "HOLD", -1: "SHORT"}


class MLSignalModel:
    """LightGBM-based signal prediction model."""

    def __init__(self, model_name: str = "signal_model"):
        self.model_name = model_name
        self.model_path = MODEL_DIR / f"{model_name}.lgb"
        self.meta_path = MODEL_DIR / f"{model_name}_meta.json"
        self.model = None
        self.meta = None

        if not ML_AVAILABLE:
            logger.error("ML libraries not available.")
            return

        # Load model if exists
        if self.model_path.exists():
            self._load_model()

    def _load_model(self):
        """Load trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            if self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    self.meta = json.load(f)
            logger.info(
                f"ML model loaded: {self.model_name} "
                f"(trained: {self.meta.get('trained_at', 'unknown')})"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _create_labels(self, df: pd.DataFrame, forward_periods: int = 6,
                       buy_threshold: float = 0.02, sell_threshold: float = -0.02) -> pd.Series:
        """
        Create target labels based on future returns.

        Args:
            df: OHLCV with indicators
            forward_periods: candles to look ahead (6 × 4h = 24h)
            buy_threshold: +2% → label as BUY
            sell_threshold: -2% → label as SHORT
        """
        future_return = df["close"].shift(-forward_periods) / df["close"] - 1

        labels = pd.Series(0, index=df.index, dtype=int)
        labels[future_return >= buy_threshold] = 1    # BUY
        labels[future_return <= sell_threshold] = -1   # SHORT

        return labels

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean feature columns from OHLCV+indicators dataframe."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        features = df[available].copy()

        # Fill NaN with 0 (indicators have warmup period)
        features = features.fillna(0)

        # Replace inf
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def train(self, dfs: dict, forward_periods: int = 6,
              buy_threshold: float = 0.02, sell_threshold: float = -0.02):
        """
        Train the model on multiple symbols' data.

        Args:
            dfs: dict of {symbol: DataFrame with indicators}
            forward_periods: candles to look ahead for labels
            buy_threshold: threshold for BUY label
            sell_threshold: threshold for SHORT label
        """
        if not ML_AVAILABLE:
            logger.error("Cannot train: ML libraries not installed")
            return None

        logger.info(f"Training ML model on {len(dfs)} symbols...")

        all_features = []
        all_labels = []

        for symbol, df in dfs.items():
            if df.empty or len(df) < 100:
                logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                continue

            features = self._prepare_features(df)
            labels = self._create_labels(df, forward_periods, buy_threshold, sell_threshold)

            # Drop rows with NaN labels (last N candles)
            valid = labels.notna() & (labels != 0) | (labels == 0)
            # Actually drop last forward_periods rows (no future data)
            valid_idx = features.index[:-forward_periods] if len(features) > forward_periods else features.index[:0]
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]

            # Drop initial warmup (first 55 rows for indicators)
            if len(features) > 55:
                features = features.iloc[55:]
                labels = labels.iloc[55:]

            all_features.append(features)
            all_labels.append(labels)

            dist = labels.value_counts().to_dict()
            logger.info(f"  {symbol}: {len(features)} samples, distribution: {dist}")

        if not all_features:
            logger.error("No training data available")
            return None

        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Total: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        # Walk-forward time series split
        tscv = TimeSeriesSplit(n_splits=5)

        best_model = None
        best_score = 0
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # LightGBM parameters
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }

            # Map labels: -1,0,1 → 0,1,2 for LightGBM
            label_map = {-1: 0, 0: 1, 1: 2}
            reverse_map = {0: -1, 1: 0, 2: 1}
            y_train_mapped = y_train.map(label_map)
            y_val_mapped = y_val.map(label_map)

            train_data = lgb.Dataset(X_train, label=y_train_mapped)
            val_data = lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            y_pred_proba = model.predict(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            score = accuracy_score(y_val_mapped, y_pred)
            fold_scores.append(score)

            if score > best_score:
                best_score = score
                best_model = model

            logger.info(f"  Fold {fold+1}: accuracy={score:.4f}")

        avg_score = np.mean(fold_scores)
        logger.info(f"Average accuracy: {avg_score:.4f} (best: {best_score:.4f})")

        # Save best model
        self.model = best_model
        joblib.dump(best_model, self.model_path)

        # Feature importance
        importance = best_model.feature_importance(importance_type="gain")
        feature_names = X.columns.tolist()
        feat_imp = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )

        # Save metadata
        self.meta = {
            "model_name": self.model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "symbols": list(dfs.keys()),
            "n_samples": len(X),
            "n_features": len(feature_names),
            "features": feature_names,
            "avg_accuracy": round(avg_score, 4),
            "best_accuracy": round(best_score, 4),
            "fold_scores": [round(s, 4) for s in fold_scores],
            "label_distribution": y.value_counts().to_dict(),
            "forward_periods": forward_periods,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "top_features": [{"name": n, "importance": round(float(i), 2)} for n, i in feat_imp[:10]],
            "label_map": {"SHORT": 0, "HOLD": 1, "BUY": 2},
            "reverse_map": {"0": -1, "1": 0, "2": 1},
        }
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

        logger.info(f"Model saved to {self.model_path}")
        logger.info("Top 10 features:")
        for name, imp in feat_imp[:10]:
            logger.info(f"  {name}: {imp:.1f}")

        return {
            "avg_accuracy": avg_score,
            "best_accuracy": best_score,
            "n_samples": len(X),
            "top_features": feat_imp[:10],
        }

    def check_signal(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Predict signal for current candle using trained model.

        Args:
            symbol: e.g. "BTC/USDT"
            df: OHLCV dataframe with indicators (from add_all_indicators)

        Returns:
            dict with signal, confidence, strategy, probabilities
        """
        if not ML_AVAILABLE or self.model is None:
            return {
                "signal": 0,
                "reason": "ML model not available or not trained",
                "strategy": "ml_signal",
                "confidence": 0.0,
            }

        if df.empty:
            return {
                "signal": 0,
                "reason": "No data",
                "strategy": "ml_signal",
                "confidence": 0.0,
            }

        features = self._prepare_features(df)
        if features.empty:
            return {
                "signal": 0,
                "reason": "No features available",
                "strategy": "ml_signal",
                "confidence": 0.0,
            }

        # Predict on latest candle
        latest_features = features.iloc[[-1]]

        try:
            proba = self.model.predict(latest_features)[0]
            # proba = [P(SHORT), P(HOLD), P(BUY)]
            pred_class = int(np.argmax(proba))

            reverse_map = {0: -1, 1: 0, 2: 1}
            signal = reverse_map[pred_class]
            confidence = float(proba[pred_class])

            # Only signal if confidence > threshold
            min_confidence = 0.45  # At least 45% confident
            if confidence < min_confidence:
                return {
                    "signal": 0,
                    "reason": (
                        f"ML low confidence: P(SHORT)={proba[0]:.2f}, "
                        f"P(HOLD)={proba[1]:.2f}, P(BUY)={proba[2]:.2f}"
                    ),
                    "strategy": "ml_signal",
                    "confidence": round(confidence, 2),
                    "probabilities": {
                        "short": round(float(proba[0]), 3),
                        "hold": round(float(proba[1]), 3),
                        "buy": round(float(proba[2]), 3),
                    },
                }

            label_name = LABEL_MAP.get(signal, "HOLD")
            reason = (
                f"ML {label_name}: confidence={confidence:.0%}, "
                f"P(SHORT)={proba[0]:.2f}, P(HOLD)={proba[1]:.2f}, P(BUY)={proba[2]:.2f}"
            )

            return {
                "signal": signal,
                "reason": reason,
                "strategy": "ml_signal",
                "confidence": round(confidence, 2),
                "probabilities": {
                    "short": round(float(proba[0]), 3),
                    "hold": round(float(proba[1]), 3),
                    "buy": round(float(proba[2]), 3),
                },
            }

        except Exception as e:
            logger.error(f"ML prediction failed for {symbol}: {e}")
            return {
                "signal": 0,
                "reason": f"ML prediction error: {str(e)[:100]}",
                "strategy": "ml_signal",
                "confidence": 0.0,
            }


def train_from_db():
    """Train ML model using historical data from database."""
    from data.fetcher import BinanceFetcher
    from utils.indicators import add_all_indicators
    from trading.signal_generator import ALPHA_CONFIGS

    fetcher = BinanceFetcher()
    dfs = {}

    print("\n=== ML Model Training ===\n")

    for symbol in ALPHA_CONFIGS:
        print(f"Fetching {symbol}...")
        try:
            df = fetcher.get_ohlcv(symbol, "4h")
            if df is not None and len(df) > 100:
                df = add_all_indicators(df)
                dfs[symbol] = df
                print(f"  {symbol}: {len(df)} candles")
            else:
                print(f"  {symbol}: skipped (insufficient data)")
        except Exception as e:
            print(f"  {symbol}: error - {e}")

    if not dfs:
        print("No data available for training!")
        return

    model = MLSignalModel()
    result = model.train(dfs)

    if result:
        print(f"\n=== Training Complete ===")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Avg Accuracy: {result['avg_accuracy']:.2%}")
        print(f"  Best Accuracy: {result['best_accuracy']:.2%}")
        print(f"\n  Top Features:")
        for name, imp in result["top_features"]:
            print(f"    {name}: {imp:.1f}")
        print(f"\n  Model saved to: models/signal_model.lgb")


def train_from_exchange():
    """Train ML model by fetching data directly from Binance API."""
    import ccxt
    from utils.indicators import add_all_indicators
    from trading.signal_generator import ALPHA_CONFIGS

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    dfs = {}
    print("\n=== ML Model Training (from exchange) ===\n")

    for symbol in ALPHA_CONFIGS:
        print(f"Fetching {symbol} from Binance...")
        try:
            candles = exchange.fetch_ohlcv(symbol, "4h", limit=1000)
            df = pd.DataFrame(
                candles,
                columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.drop(columns=["timestamp_ms"]).set_index("timestamp")
            df = add_all_indicators(df)
            dfs[symbol] = df
            print(f"  {symbol}: {len(df)} candles")
        except Exception as e:
            print(f"  {symbol}: error - {e}")

    if not dfs:
        print("No data available for training!")
        return

    model = MLSignalModel()
    result = model.train(dfs)

    if result:
        print(f"\n=== Training Complete ===")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Avg Accuracy: {result['avg_accuracy']:.2%}")
        print(f"  Best Accuracy: {result['best_accuracy']:.2%}")
        print(f"\n  Top Features:")
        for name, imp in result["top_features"]:
            print(f"    {name}: {imp:.1f}")
        print(f"\n  Model saved to: models/signal_model.lgb")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML Signal Model")
    parser.add_argument("--train", action="store_true", help="Train from database")
    parser.add_argument("--train-exchange", action="store_true", help="Train from Binance API")
    args = parser.parse_args()

    if args.train:
        train_from_db()
    elif args.train_exchange:
        train_from_exchange()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m strategies.ml_model --train           # Train from DB")
        print("  python -m strategies.ml_model --train-exchange   # Train from Binance API")
