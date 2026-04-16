"""
Strategy Parameter Optimizer
==============================
Grid search for optimal parameters on train set,
then validate on test set to avoid overfitting.

Usage:
    from strategies.optimizer import optimize_strategy
    best_params, results = optimize_strategy("momentum_reversal", df_train)
"""

import itertools
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from backtest import BacktestEngine
from utils.indicators import add_all_indicators
from config.settings import SLIPPAGE_PCT


# ═══════════════════════════════════════════
# PARAMETER GRIDS
# ═══════════════════════════════════════════

PARAM_GRIDS = {
    "momentum_reversal": {
        "roc_threshold": [-5, -8, -10, -12, -15],
        "roc_exit": [1, 2, 3, 5],
        "support_buffer": [0.02, 0.05, 0.08, 0.12],
    },
    "volatility_breakout": {
        "atr_multiplier": [1.0, 1.5, 2.0, 2.5],
        "volume_threshold": [1.2, 1.5, 2.0, 2.5],
        "holding_periods": [3, 6, 9, 12],
    },
    "composite": {
        "w_mr": [0.2, 0.3, 0.4, 0.5],
        "w_vb": [0.1, 0.2, 0.3],
        "threshold": [0.15, 0.25, 0.35, 0.45],
    },
    "momentum_reversal_mtf": {
        "roc_threshold": [-5, -8, -10, -12, -15],
        "roc_exit": [1, 2, 3, 5],
        "support_buffer": [0.02, 0.05, 0.08, 0.12],
    },
}

STOP_LOSS_GRID = [0.03, 0.04, 0.05, 0.07]


# ═══════════════════════════════════════════
# STRATEGY SIGNAL GENERATORS (parameterized)
# ═══════════════════════════════════════════

def _signal_momentum_reversal(df, roc_threshold, roc_exit, support_buffer):
    """Momentum reversal with custom parameters."""
    near_support = df["price_position"] < support_buffer + 0.1
    near_resistance = df["price_position"] > 1 - support_buffer - 0.1

    oversold = (df["roc_10"] < roc_threshold) & near_support
    overbought = (df["roc_10"] > abs(roc_threshold)) & near_resistance

    signal = pd.Series(0, index=df.index, dtype=int)
    position = 0

    for i in range(len(df)):
        if oversold.iloc[i] and position <= 0:
            position = 1
        elif overbought.iloc[i] and position >= 0:
            position = -1
        elif position == 1 and df["roc_10"].iloc[i] > roc_exit:
            position = 0
        elif position == -1 and df["roc_10"].iloc[i] < -roc_exit:
            position = 0
        signal.iloc[i] = position

    return signal


def _signal_volatility_breakout(df, atr_multiplier, volume_threshold, holding_periods):
    """Volatility breakout with custom parameters."""
    upper_band = df["sma_20"] + atr_multiplier * df["atr"]
    lower_band = df["sma_20"] - atr_multiplier * df["atr"]
    vol_spike = df["volume_ratio"] > volume_threshold

    breakout_up = (df["close"] > upper_band) & vol_spike
    breakout_down = (df["close"] < lower_band) & vol_spike

    signal = pd.Series(0, index=df.index, dtype=int)
    hold_counter = 0
    position = 0

    for i in range(len(df)):
        if hold_counter > 0:
            hold_counter -= 1
            signal.iloc[i] = position
            if hold_counter == 0:
                position = 0
            continue

        if breakout_up.iloc[i]:
            position = 1
            hold_counter = holding_periods
        elif breakout_down.iloc[i]:
            position = -1
            hold_counter = holding_periods

        signal.iloc[i] = position

    return signal


def _signal_composite(df, w_mr, w_vb, threshold):
    """Composite with custom weights."""
    w_tf = max(0, 1.0 - w_mr - w_vb)

    s_mr = _signal_momentum_reversal(df, roc_threshold=-10, roc_exit=2, support_buffer=0.05)
    s_vb = _signal_volatility_breakout(df, atr_multiplier=1.5, volume_threshold=1.5, holding_periods=6)

    # Simple trend signal for the remaining weight
    ema_cross = (df["ema_9"] > df["ema_21"]).astype(int) * 2 - 1
    s_tf = ema_cross

    score = w_mr * s_mr + w_vb * s_vb + w_tf * s_tf

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[score > threshold] = 1
    signal[score < -threshold] = -1

    return signal


def _signal_momentum_reversal_mtf(df, roc_threshold, roc_exit, support_buffer):
    """Momentum reversal filtered by daily trend bias (MTF)."""
    # Get base signals
    base_signals = _signal_momentum_reversal(df, roc_threshold, roc_exit, support_buffer)

    # Compute daily trend from longer-period indicators
    bullish = (df["ema_50"] > df["sma_200"]) & (df["close"] > df["ema_50"])
    bearish = (df["ema_50"] < df["sma_200"]) & (df["close"] < df["ema_50"])

    trend = pd.Series(0, index=df.index, dtype=int)
    trend[bullish] = 1
    trend[bearish] = -1

    # Filter: bullish → only longs, bearish → only shorts, neutral → all
    filtered = base_signals.copy()
    filtered[(trend == 1) & (base_signals == -1)] = 0
    filtered[(trend == -1) & (base_signals == 1)] = 0

    return filtered


SIGNAL_FUNCS = {
    "momentum_reversal": _signal_momentum_reversal,
    "volatility_breakout": _signal_volatility_breakout,
    "composite": _signal_composite,
    "momentum_reversal_mtf": _signal_momentum_reversal_mtf,
}


# ═══════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════

def optimize_strategy(
    strategy_key: str,
    df_train: pd.DataFrame,
    initial_capital: float = 500,
    fee: float = 0.001,
    top_n: int = 5,
) -> tuple:
    """
    Grid search for best parameters on train set.

    Args:
        strategy_key: One of 'momentum_reversal', 'volatility_breakout', 'composite'
        df_train: Training data with indicators
        initial_capital: Starting capital
        fee: Trading fee
        top_n: Number of top results to return

    Returns:
        (best_params_dict, results_dataframe)
    """
    param_grid = PARAM_GRIDS[strategy_key]
    signal_func = SIGNAL_FUNCS[strategy_key]

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    # Add stop-loss to search
    all_combos = []
    for combo in combinations:
        for sl in STOP_LOSS_GRID:
            all_combos.append((*combo, sl))

    total = len(all_combos)
    logger.info(f"Optimizing {strategy_key}: {total} parameter combinations")

    results = []

    for combo in tqdm(all_combos, desc=f"Optimizing {strategy_key}"):
        params = dict(zip(param_names, combo[:-1]))
        sl = combo[-1]

        try:
            signals = signal_func(df_train, **params)

            bt = BacktestEngine(
                df_train,
                initial_capital=initial_capital,
                fee=fee,
                slippage_pct=SLIPPAGE_PCT,
                stop_loss_pct=sl,
            )
            bt.run(signals)
            metrics = bt.get_metrics()

            result = {**params, "stop_loss": sl}
            result["sharpe"] = metrics["sharpe_ratio"]
            result["return_pct"] = metrics["total_return_pct"]
            result["max_dd_pct"] = metrics["max_drawdown_pct"]
            result["win_rate"] = metrics["win_rate_pct"]
            result["profit_factor"] = metrics["profit_factor"]
            result["trades"] = metrics["total_trades"]
            result["final_equity"] = metrics["final_equity"]

            results.append(result)
        except Exception as e:
            continue

    if not results:
        logger.error(f"No valid results for {strategy_key}")
        return None, None

    results_df = pd.DataFrame(results)

    # Filter: minimum 5 trades, max drawdown > -20%
    filtered = results_df[
        (results_df["trades"] >= 5) &
        (results_df["max_dd_pct"] > -20)
    ]

    if len(filtered) == 0:
        # Relax constraints
        filtered = results_df[results_df["trades"] >= 3]
        logger.warning("No results with DD < 20%, relaxing constraints")

    # Sort by Sharpe
    filtered = filtered.sort_values("sharpe", ascending=False)
    top = filtered.head(top_n)

    best = top.iloc[0].to_dict()

    logger.info(
        f"Best {strategy_key}: Sharpe={best['sharpe']:.3f}, "
        f"Return={best['return_pct']:.2f}%, "
        f"MaxDD={best['max_dd_pct']:.2f}%"
    )

    return best, filtered


def validate_on_test(
    strategy_key: str,
    best_params: dict,
    df_test: pd.DataFrame,
    initial_capital: float = 500,
    fee: float = 0.001,
) -> dict:
    """
    Validate optimized parameters on out-of-sample test set.

    Returns:
        metrics dict
    """
    signal_func = SIGNAL_FUNCS[strategy_key]
    param_grid = PARAM_GRIDS[strategy_key]
    param_names = list(param_grid.keys())

    # Extract strategy params (exclude stop_loss)
    strat_params = {k: best_params[k] for k in param_names}
    sl = best_params["stop_loss"]

    signals = signal_func(df_test, **strat_params)

    bt = BacktestEngine(
        df_test,
        initial_capital=initial_capital,
        fee=fee,
        slippage_pct=SLIPPAGE_PCT,
        stop_loss_pct=sl,
    )
    bt.run(signals)
    metrics = bt.get_metrics()

    return metrics, bt
