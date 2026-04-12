"""
Alpha Comparison Runner
========================
Backtests all 5 strategies on multiple coins and timeframes,
then produces a comparison report.

Usage:
    python -m strategies.run_backtest
    python -m strategies.run_backtest --symbol BTC/USDT --timeframe 4h
    python -m strategies.run_backtest --all
"""

import argparse
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from sqlalchemy import text
from loguru import logger

from config.settings import COIN_UNIVERSE, DATABASE_URL, LOG_FILE, SLIPPAGE_PCT
from data.models import engine
from backtest import BacktestEngine, walk_forward_split
from strategies.technical_alphas import STRATEGIES
from utils.indicators import add_all_indicators

logger.add(LOG_FILE, rotation="10 MB", level="INFO")


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


def run_single(symbol: str, timeframe: str, strategy_name: str) -> dict:
    """Run one strategy on one symbol/timeframe with walk-forward validation."""
    df = load_data(symbol, timeframe)
    if len(df) < 200:
        logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
        return None

    df = add_all_indicators(df)
    df = df.dropna()

    # Walk-forward split
    train_df, test_df = walk_forward_split(df, train_ratio=0.7)

    strategy = STRATEGIES[strategy_name]
    signal_func = strategy["func"]

    # Generate signals on FULL data (indicators need history)
    # But evaluate only on test set
    signals = signal_func(df)
    test_signals = signals.reindex(test_df.index)

    # Backtest on test set only
    bt = BacktestEngine(
        test_df,
        initial_capital=500,
        fee=0.001,
        slippage_pct=SLIPPAGE_PCT,
        stop_loss_pct=0.05,
    )
    bt.run(test_signals)
    metrics = bt.get_metrics()

    metrics["symbol"] = symbol
    metrics["timeframe"] = timeframe
    metrics["strategy"] = strategy["name"]
    metrics["strategy_key"] = strategy_name

    return metrics


def run_comparison(
    symbols: list = None,
    timeframe: str = "4h",
    show_plots: bool = False,
):
    """Run all strategies on given symbols and compare."""
    symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]

    all_results = []

    print(f"\nRunning {len(STRATEGIES)} strategies on {len(symbols)} coins ({timeframe})")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'─' * 70}")
        print(f"  {symbol}")
        print(f"{'─' * 70}")

        for strat_key, strat_info in STRATEGIES.items():
            try:
                result = run_single(symbol, timeframe, strat_key)
                if result:
                    all_results.append(result)
                    sharpe = result["sharpe_ratio"]
                    ret = result["total_return_pct"]
                    dd = result["max_drawdown_pct"]
                    trades = result["total_trades"]
                    print(
                        f"  {strat_info['name']:<35} "
                        f"Sharpe={sharpe:>6.3f}  "
                        f"Return={ret:>+7.2f}%  "
                        f"MaxDD={dd:>7.2f}%  "
                        f"Trades={trades:>4}"
                    )
            except Exception as e:
                logger.error(f"Failed {symbol} {strat_key}: {e}")
                print(f"  {strat_info['name']:<35} ERROR: {e}")

    if not all_results:
        print("\nNo results to compare.")
        return

    # Build comparison DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary by strategy
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON SUMMARY (averaged across coins)")
    print("=" * 70)

    summary = results_df.groupby("strategy").agg({
        "sharpe_ratio": "mean",
        "total_return_pct": "mean",
        "max_drawdown_pct": "mean",
        "win_rate_pct": "mean",
        "profit_factor": "mean",
        "total_trades": "mean",
        "alpha_pct": "mean",
    }).round(3)

    summary = summary.sort_values("sharpe_ratio", ascending=False)

    print(summary.to_string())

    # Best strategy per coin
    print("\n" + "=" * 70)
    print("  BEST STRATEGY PER COIN (by Sharpe)")
    print("=" * 70)

    for symbol in symbols:
        coin_results = results_df[results_df["symbol"] == symbol]
        if len(coin_results) == 0:
            continue
        best = coin_results.loc[coin_results["sharpe_ratio"].idxmax()]
        print(
            f"  {symbol:<12} → {best['strategy']:<35} "
            f"Sharpe={best['sharpe_ratio']:.3f}"
        )

    # Strategies meeting minimum criteria
    print("\n" + "=" * 70)
    print("  ALPHA CANDIDATES (Sharpe > 0.5 on test set)")
    print("=" * 70)

    good = results_df[results_df["sharpe_ratio"] > 0.5]
    if len(good) > 0:
        for _, row in good.iterrows():
            print(
                f"  {row['symbol']:<12} {row['strategy']:<35} "
                f"Sharpe={row['sharpe_ratio']:.3f}  "
                f"Return={row['total_return_pct']:+.2f}%"
            )
    else:
        print("  None found. Consider adjusting parameters or strategies.")

    # Save results
    output_file = "backtest_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha - Strategy Backtest Runner")
    parser.add_argument("--symbol", type=str, help="Single symbol, e.g. BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe (default: 4h)")
    parser.add_argument("--strategy", type=str, help="Single strategy key")
    parser.add_argument("--all", action="store_true", help="Run on all coins in universe")
    parser.add_argument("--top", type=int, default=5, help="Top N coins by volume (default: 5)")

    args = parser.parse_args()

    if args.symbol and args.strategy:
        # Single run with detailed report
        result = run_single(args.symbol, args.timeframe, args.strategy)
        if result:
            # Re-run to show report and plot
            df = load_data(args.symbol, args.timeframe)
            df = add_all_indicators(df).dropna()
            _, test_df = walk_forward_split(df, 0.7)
            signals = STRATEGIES[args.strategy]["func"](df).reindex(test_df.index)

            bt = BacktestEngine(test_df, initial_capital=500, fee=0.001, stop_loss_pct=0.05)
            bt.run(signals)
            bt.print_report()
            bt.plot_equity(f"{args.symbol} — {STRATEGIES[args.strategy]['name']}")
    elif args.all:
        symbols = COIN_UNIVERSE
        run_comparison(symbols=symbols, timeframe=args.timeframe)
    else:
        # Default: top coins comparison
        top_coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
        if args.symbol:
            top_coins = [args.symbol]
        run_comparison(symbols=top_coins, timeframe=args.timeframe)


if __name__ == "__main__":
    main()
