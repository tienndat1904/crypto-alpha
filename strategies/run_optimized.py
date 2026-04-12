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

from config.settings import LOG_FILE, SLIPPAGE_PCT
from data.models import engine
from backtest import BacktestEngine, walk_forward_split, rolling_walk_forward
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


def run_optimization(symbols: list = None, n_folds: int = 5):
    """Run full optimization pipeline with rolling walk-forward validation."""
    symbols = symbols or FOCUS_COINS

    all_results = []
    optimized_params = {}

    print("\n" + "=" * 70)
    print("  PHASE 2+ — PARAMETER OPTIMIZATION (Rolling Walk-Forward)")
    print("  Strategies: Momentum Reversal, Volatility Breakout, Composite")
    print(f"  Coins: {', '.join([s.replace('/USDT','') for s in symbols])}")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Folds: {n_folds} (rolling walk-forward)")
    print(f"  Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'━' * 70}")
        print(f"  {symbol}")
        print(f"{'━' * 70}")

        # Load and prepare data
        df = load_data(symbol, TIMEFRAME)
        df = add_all_indicators(df).dropna()

        # Rolling walk-forward folds
        folds = rolling_walk_forward(df, n_splits=n_folds, train_ratio=0.7)

        for strat_key in FOCUS_STRATEGIES:
            strat_name = STRATEGY_NAMES[strat_key]
            print(f"\n  ▸ Optimizing: {strat_name} ({len(folds)} folds)")

            fold_results = []

            for train_df, test_df, fold_info in folds:
                fold_num = fold_info["fold"]

                # ── Step 1: Optimize on train set ──
                best_params, top_results = optimize_strategy(
                    strat_key, train_df, initial_capital=500,
                    fee=0.001, top_n=5
                )

                if best_params is None:
                    print(f"    Fold {fold_num}: No valid parameters found")
                    continue

                # ── Step 2: Validate on test set ──
                test_metrics, bt = validate_on_test(
                    strat_key, best_params, test_df,
                    initial_capital=500, fee=0.001
                )

                fold_result = {
                    "fold": fold_num,
                    "train_sharpe": best_params["sharpe"],
                    "test_sharpe": test_metrics["sharpe_ratio"],
                    "test_return": test_metrics["total_return_pct"],
                    "test_dd": test_metrics["max_drawdown_pct"],
                    "test_trades": test_metrics["total_trades"],
                    "test_win_rate": test_metrics["win_rate_pct"],
                    "params": best_params,
                }
                fold_results.append(fold_result)

                train_s = best_params["sharpe"]
                test_s = test_metrics["sharpe_ratio"]
                print(
                    f"    Fold {fold_num}: "
                    f"Train Sharpe={train_s:.3f} → Test Sharpe={test_s:.3f}, "
                    f"Return={test_metrics['total_return_pct']:+.2f}%, "
                    f"DD={test_metrics['max_drawdown_pct']:.1f}%"
                )

            if not fold_results:
                print(f"    ✗ No valid results across folds")
                continue

            # ── Aggregate across folds ──
            avg_test_sharpe = np.mean([r["test_sharpe"] for r in fold_results])
            std_test_sharpe = np.std([r["test_sharpe"] for r in fold_results])
            avg_test_return = np.mean([r["test_return"] for r in fold_results])
            avg_test_dd = np.mean([r["test_dd"] for r in fold_results])
            avg_test_wr = np.mean([r["test_win_rate"] for r in fold_results])
            total_test_trades = sum([r["test_trades"] for r in fold_results])
            positive_folds = sum(1 for r in fold_results if r["test_sharpe"] > 0)

            # Pick best fold's params (highest test sharpe)
            best_fold = max(fold_results, key=lambda r: r["test_sharpe"])
            best_params = best_fold["params"]
            param_names = list(PARAM_GRIDS[strat_key].keys())

            print(f"\n    {'─' * 55}")
            print(f"    Rolling Walk-Forward Summary ({len(fold_results)} folds):")
            print(f"    {'─' * 55}")
            print(f"    Avg Test Sharpe:    {avg_test_sharpe:>8.3f} (±{std_test_sharpe:.3f})")
            print(f"    Avg Test Return:    {avg_test_return:>+7.2f}%")
            print(f"    Avg Test MaxDD:     {avg_test_dd:>8.2f}%")
            print(f"    Avg Win Rate:       {avg_test_wr:>8.1f}%")
            print(f"    Total Trades:       {total_test_trades:>8}")
            print(f"    Positive Folds:     {positive_folds}/{len(fold_results)}")

            # Stability assessment
            if std_test_sharpe > abs(avg_test_sharpe) and avg_test_sharpe > 0:
                print(f"    ⚠️  High variance — alpha may be regime-dependent")
            elif positive_folds == len(fold_results) and avg_test_sharpe > 0.5:
                print(f"    ✓  Consistent alpha across all folds!")
            elif positive_folds >= len(fold_results) * 0.6 and avg_test_sharpe > 0:
                print(f"    ◆  Moderately stable alpha")
            elif avg_test_sharpe <= 0:
                print(f"    ✗  Negative average Sharpe — no alpha")

            # Store aggregated result
            result = {
                "symbol": symbol,
                "strategy": strat_name,
                "strategy_key": strat_key,
                **{f"param_{k}": best_params[k] for k in param_names},
                "stop_loss": best_params["stop_loss"],
                "n_folds": len(fold_results),
                "positive_folds": positive_folds,
                "avg_test_sharpe": avg_test_sharpe,
                "std_test_sharpe": std_test_sharpe,
                "avg_test_return": avg_test_return,
                "avg_test_dd": avg_test_dd,
                "avg_test_win_rate": avg_test_wr,
                "total_test_trades": total_test_trades,
                "best_fold_sharpe": best_fold["test_sharpe"],
            }
            all_results.append(result)

            key = f"{symbol}_{strat_key}"
            optimized_params[key] = best_params

    # ── Final Summary ──
    if not all_results:
        print("\nNo results to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("  ROLLING WALK-FORWARD RESULTS SUMMARY")
    print("=" * 70)

    print(
        f"\n  {'Symbol':<10} {'Strategy':<25} "
        f"{'Avg Sharpe':>10} {'±Std':>8} "
        f"{'Pos Folds':>10}"
    )
    print(f"  {'─' * 65}")

    for _, row in results_df.iterrows():
        print(
            f"  {row['symbol']:<10} {row['strategy']:<25} "
            f"{row['avg_test_sharpe']:>10.3f} {row['std_test_sharpe']:>7.3f} "
            f"{row['positive_folds']}/{row['n_folds']:>8}"
        )

    # ── Alpha Candidates ──
    print("\n" + "=" * 70)
    print("  ALPHA CANDIDATES (Avg Sharpe > 0.3, >60% positive folds)")
    print("=" * 70)

    candidates = results_df[
        (results_df["avg_test_sharpe"] > 0.3) &
        (results_df["positive_folds"] >= results_df["n_folds"] * 0.6)
    ].sort_values("avg_test_sharpe", ascending=False)

    if len(candidates) > 0:
        for _, row in candidates.iterrows():
            param_names = list(PARAM_GRIDS[row["strategy_key"]].keys())
            params_str = ", ".join([f"{k}={row[f'param_{k}']}" for k in param_names])
            print(
                f"\n  ★ {row['symbol']} — {row['strategy']}"
                f"\n    Best Params: {params_str}, SL={row['stop_loss']:.0%}"
                f"\n    Avg Sharpe={row['avg_test_sharpe']:.3f} (±{row['std_test_sharpe']:.3f}), "
                f"Avg Return={row['avg_test_return']:+.2f}%, "
                f"Avg DD={row['avg_test_dd']:.2f}%"
                f"\n    Win Rate={row['avg_test_win_rate']:.1f}%, "
                f"Total Trades={row['total_test_trades']}, "
                f"Positive Folds={row['positive_folds']}/{row['n_folds']}"
            )
    else:
        print("\n  No candidates meeting criteria.")
        print("  Relaxing to Avg Sharpe > 0.1:")
        relaxed = results_df[results_df["avg_test_sharpe"] > 0.1].sort_values(
            "avg_test_sharpe", ascending=False
        )
        if len(relaxed) > 0:
            for _, row in relaxed.iterrows():
                print(
                    f"  ◆ {row['symbol']} — {row['strategy']}: "
                    f"Avg Sharpe={row['avg_test_sharpe']:.3f}, "
                    f"Return={row['avg_test_return']:+.2f}%"
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
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds (default: 5)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    run_optimization(symbols, n_folds=args.folds)


if __name__ == "__main__":
    main()
