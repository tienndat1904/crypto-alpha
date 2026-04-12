"""
Optimized Backtest Runner
==========================
Optimizes 3 selected strategies on BTC, SOL, XRP,
then validates on test set and compares before/after.

Usage:
    python -m strategies.run_optimized
    python -m strategies.run_optimized --symbol BTC/USDT
"""

import argparse
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from sqlalchemy import text
from loguru import logger

from config.settings import LOG_FILE
from data.models import engine
from backtest import BacktestEngine, walk_forward_split
from strategies.optimizer import (
    optimize_strategy,
    validate_on_test,
    PARAM_GRIDS,
)
from utils.indicators import add_all_indicators

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# Focus coins based on Phase 2 results
FOCUS_COINS = ["BTC/USDT", "SOL/USDT", "XRP/USDT"]
FOCUS_STRATEGIES = ["momentum_reversal", "volatility_breakout", "composite"]
TIMEFRAME = "4h"

STRATEGY_NAMES = {
    "momentum_reversal": "Momentum Reversal",
    "volatility_breakout": "Volatility Breakout",
    "composite": "Composite Signal",
}


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV from MySQL."""
    query = text(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv WHERE symbol = :symbol AND timeframe = :timeframe "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "timeframe": timeframe})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def run_optimization(symbols: list = None):
    """Run full optimization pipeline."""
    symbols = symbols or FOCUS_COINS

    all_results = []
    optimized_params = {}

    print("\n" + "=" * 70)
    print("  PHASE 2 — PARAMETER OPTIMIZATION")
    print("  Strategies: Momentum Reversal, Volatility Breakout, Composite")
    print(f"  Coins: {', '.join([s.replace('/USDT','') for s in symbols])}")
    print(f"  Timeframe: {TIMEFRAME}")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'━' * 70}")
        print(f"  {symbol}")
        print(f"{'━' * 70}")

        # Load and prepare data
        df = load_data(symbol, TIMEFRAME)
        df = add_all_indicators(df).dropna()
        train_df, test_df = walk_forward_split(df, train_ratio=0.7)

        for strat_key in FOCUS_STRATEGIES:
            strat_name = STRATEGY_NAMES[strat_key]
            print(f"\n  ▸ Optimizing: {strat_name}")

            # ── Step 1: Optimize on train set ──
            best_params, top_results = optimize_strategy(
                strat_key, train_df, initial_capital=500, fee=0.001, top_n=5
            )

            if best_params is None:
                print(f"    ✗ No valid parameters found")
                continue

            # Show top 3 parameter sets
            print(f"\n    Top 3 parameter sets (on train):")
            param_names = list(PARAM_GRIDS[strat_key].keys())
            for idx, row in top_results.head(3).iterrows():
                params_str = ", ".join([f"{k}={row[k]}" for k in param_names])
                print(
                    f"      {params_str}, SL={row['stop_loss']:.0%} "
                    f"→ Sharpe={row['sharpe']:.3f}, "
                    f"Return={row['return_pct']:+.2f}%, "
                    f"DD={row['max_dd_pct']:.1f}%"
                )

            # ── Step 2: Validate on test set ──
            print(f"\n    Validating best params on TEST set...")
            test_metrics, bt = validate_on_test(
                strat_key, best_params, test_df, initial_capital=500, fee=0.001
            )

            # Store results
            result = {
                "symbol": symbol,
                "strategy": strat_name,
                "strategy_key": strat_key,
                **{f"param_{k}": best_params[k] for k in param_names},
                "stop_loss": best_params["stop_loss"],
                "train_sharpe": best_params["sharpe"],
                "train_return": best_params["return_pct"],
                "train_dd": best_params["max_dd_pct"],
                "test_sharpe": test_metrics["sharpe_ratio"],
                "test_return": test_metrics["total_return_pct"],
                "test_dd": test_metrics["max_drawdown_pct"],
                "test_win_rate": test_metrics["win_rate_pct"],
                "test_profit_factor": test_metrics["profit_factor"],
                "test_trades": test_metrics["total_trades"],
                "test_final_equity": test_metrics["final_equity"],
                "test_fees": test_metrics["total_fees_usd"],
            }
            all_results.append(result)

            # Save optimized params
            key = f"{symbol}_{strat_key}"
            optimized_params[key] = best_params

            # Print comparison
            train_s = best_params["sharpe"]
            test_s = test_metrics["sharpe_ratio"]
            degradation = ((test_s - train_s) / abs(train_s) * 100) if train_s != 0 else 0

            print(f"\n    {'Metric':<20} {'Train':>10} {'Test':>10} {'Δ':>10}")
            print(f"    {'─' * 50}")
            print(f"    {'Sharpe':<20} {train_s:>10.3f} {test_s:>10.3f} {degradation:>+9.1f}%")
            print(f"    {'Return':<20} {best_params['return_pct']:>+9.2f}% {test_metrics['total_return_pct']:>+9.2f}%")
            print(f"    {'Max Drawdown':<20} {best_params['max_dd_pct']:>9.2f}% {test_metrics['max_drawdown_pct']:>9.2f}%")
            print(f"    {'Win Rate':<20} {'':>10} {test_metrics['win_rate_pct']:>9.1f}%")
            print(f"    {'Trades':<20} {'':>10} {test_metrics['total_trades']:>10}")
            print(f"    {'Final Equity':<20} {'':>10} ${test_metrics['final_equity']:>9.2f}")

            # Overfitting warning
            if degradation < -50:
                print(f"\n    ⚠️  WARNING: Sharpe degraded {degradation:.0f}% — possible overfitting!")
            elif test_s > 0:
                print(f"\n    ✓  Positive Sharpe on test set — alpha candidate confirmed!")

    # ── Final Summary ──
    if not all_results:
        print("\nNo results to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS — OPTIMIZED vs DEFAULT")
    print("=" * 70)

    # Default results from Phase 2 for comparison
    defaults = {
        ("BTC/USDT", "Momentum Reversal"): {"sharpe": -1.638, "return": -24.37},
        ("BTC/USDT", "Volatility Breakout"): {"sharpe": 0.648, "return": 10.32},
        ("BTC/USDT", "Composite Signal"): {"sharpe": -1.955, "return": -38.76},
        ("SOL/USDT", "Momentum Reversal"): {"sharpe": 0.873, "return": 19.23},
        ("SOL/USDT", "Volatility Breakout"): {"sharpe": -1.538, "return": -52.80},
        ("SOL/USDT", "Composite Signal"): {"sharpe": -0.936, "return": -27.45},
        ("XRP/USDT", "Momentum Reversal"): {"sharpe": 1.206, "return": 26.59},
        ("XRP/USDT", "Volatility Breakout"): {"sharpe": -0.885, "return": -25.33},
        ("XRP/USDT", "Composite Signal"): {"sharpe": 0.958, "return": 21.74},
    }

    print(f"\n  {'Symbol':<10} {'Strategy':<25} {'Default':>10} {'Optimized':>10} {'Improved':>10}")
    print(f"  {'─' * 65}")

    for _, row in results_df.iterrows():
        key = (row["symbol"], row["strategy"])
        default = defaults.get(key, {"sharpe": 0, "return": 0})
        default_s = default["sharpe"]
        opt_s = row["test_sharpe"]
        improved = "✓ YES" if opt_s > default_s else "✗ NO"

        print(
            f"  {row['symbol']:<10} {row['strategy']:<25} "
            f"{default_s:>10.3f} {opt_s:>10.3f} {improved:>10}"
        )

    # ── Alpha Candidates ──
    print("\n" + "=" * 70)
    print("  FINAL ALPHA CANDIDATES (Test Sharpe > 0.5, DD > -15%)")
    print("=" * 70)

    candidates = results_df[
        (results_df["test_sharpe"] > 0.5) &
        (results_df["test_dd"] > -15)
    ].sort_values("test_sharpe", ascending=False)

    if len(candidates) > 0:
        for _, row in candidates.iterrows():
            param_names = list(PARAM_GRIDS[row["strategy_key"]].keys())
            params_str = ", ".join([f"{k}={row[f'param_{k}']}" for k in param_names])
            print(
                f"\n  ★ {row['symbol']} — {row['strategy']}"
                f"\n    Params: {params_str}, SL={row['stop_loss']:.0%}"
                f"\n    Test: Sharpe={row['test_sharpe']:.3f}, "
                f"Return={row['test_return']:+.2f}%, "
                f"DD={row['test_dd']:.2f}%, "
                f"WinRate={row['test_win_rate']:.1f}%, "
                f"Trades={row['test_trades']}"
                f"\n    Final Equity: ${row['test_final_equity']:.2f} "
                f"(from $500)"
            )
    else:
        print("\n  No candidates meeting strict criteria.")
        print("  Relaxing to Sharpe > 0.3:")
        relaxed = results_df[results_df["test_sharpe"] > 0.3].sort_values(
            "test_sharpe", ascending=False
        )
        if len(relaxed) > 0:
            for _, row in relaxed.iterrows():
                print(
                    f"  ◆ {row['symbol']} — {row['strategy']}: "
                    f"Sharpe={row['test_sharpe']:.3f}, "
                    f"Return={row['test_return']:+.2f}%"
                )
        else:
            print("  No candidates. Consider different strategies or timeframes.")

    # Save
    results_df.to_csv("optimized_results.csv", index=False)
    print(f"\nResults saved to: optimized_results.csv")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha — Optimized Backtest")
    parser.add_argument("--symbol", type=str, help="Single symbol")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    run_optimization(symbols)


if __name__ == "__main__":
    main()
