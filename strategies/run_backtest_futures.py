"""
Futures Backtest Runner (Long + Short, Leveraged)
==================================================
Backtests all strategies on multiple coins using LONG+SHORT signals
with leverage, lower fees (futures), and tighter stop losses.

Usage:
    python -m strategies.run_backtest_futures
    python -m strategies.run_backtest_futures --symbol BTC/USDT --timeframe 4h
    python -m strategies.run_backtest_futures --all
"""

import argparse
import json
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from sqlalchemy import text
from loguru import logger

from config.settings import (
    COIN_UNIVERSE, DATABASE_URL, LOG_FILE, SLIPPAGE_PCT,
    FUTURES_LEVERAGE, FUTURES_FEE,
)
from data.models import engine
from backtest import BacktestEngine, walk_forward_split
from strategies.technical_alphas import STRATEGIES  # Full long+short strategies
from utils.indicators import add_all_indicators

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# Futures-specific configs
FUTURES_STOP_LOSS = 0.02  # 2% (tighter for leverage)


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


def run_single(symbol: str, timeframe: str, strategy_name: str) -> tuple:
    """Run one strategy on one symbol/timeframe with walk-forward validation.

    Returns:
        (metrics_dict, BacktestEngine) or (None, None) if not enough data.
    """
    df = load_data(symbol, timeframe)
    if len(df) < 200:
        logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
        return None, None

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

    # Backtest on test set (futures: long+short, leveraged)
    bt = BacktestEngine(
        test_df,
        initial_capital=500,
        fee=FUTURES_FEE,
        slippage_pct=SLIPPAGE_PCT,
        stop_loss_pct=FUTURES_STOP_LOSS,
        leverage=FUTURES_LEVERAGE,
    )
    bt.run(test_signals)
    metrics = bt.get_metrics()

    metrics["symbol"] = symbol
    metrics["timeframe"] = timeframe
    metrics["strategy"] = strategy["name"]
    metrics["strategy_key"] = strategy_name
    metrics["split_date"] = str(test_df.index[0].date())
    metrics["leverage"] = FUTURES_LEVERAGE

    return metrics, bt


def run_comparison(
    symbols: list = None,
    timeframe: str = "4h",
):
    """Run all strategies on given symbols and compare."""
    symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]

    all_results = []
    equity_curves = {}

    print(f"\n[FUTURES BACKTEST - Long+Short, {FUTURES_LEVERAGE}x Leverage]")
    print(f"Running {len(STRATEGIES)} strategies on {len(symbols)} coins ({timeframe})")
    print(f"Fee: {FUTURES_FEE*100:.2f}% | Stop Loss: {FUTURES_STOP_LOSS*100:.1f}%")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'-' * 70}")
        print(f"  {symbol}")
        print(f"{'-' * 70}")

        for strat_key, strat_info in STRATEGIES.items():
            try:
                result, bt = run_single(symbol, timeframe, strat_key)
                if result and bt is not None:
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

                    # Collect equity curve data
                    res_df = bt.results
                    returns = res_df["strategy_return"]
                    rolling_sharpe = (
                        returns.rolling(window=126, min_periods=30).mean()
                        / returns.rolling(window=126, min_periods=30).std()
                    ) * np.sqrt(365 * 6)
                    turnover = res_df["trade"].abs().rolling(window=126, min_periods=30).mean()

                    curve_key = f"{symbol}|{strat_key}"
                    equity_curves[curve_key] = {
                        "symbol": symbol,
                        "strategy": strat_info["name"],
                        "leverage": FUTURES_LEVERAGE,
                        "dates": [str(d) for d in res_df.index.tolist()],
                        "equity": res_df["equity"].tolist(),
                        "buy_hold": res_df["buy_hold_equity"].tolist(),
                        "drawdown": res_df["drawdown"].tolist(),
                        "cum_pnl": (res_df["equity"] - bt.initial_capital).tolist(),
                        "rolling_sharpe": rolling_sharpe.fillna(0).tolist(),
                        "rolling_turnover": turnover.fillna(0).tolist(),
                        "split_date": result.get("split_date", ""),
                    }
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
    print(f"  FUTURES STRATEGY COMPARISON ({FUTURES_LEVERAGE}x, averaged across coins)")
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
            f"  {symbol:<12} -> {best['strategy']:<35} "
            f"Sharpe={best['sharpe_ratio']:.3f}"
        )

    # Alpha candidates
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
        print("  None found.")

    # Save results
    output_file = "backtest_futures_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save equity curves
    if equity_curves:
        equity_file = "backtest_futures_equity.json"
        with open(equity_file, "w") as f:
            json.dump(equity_curves, f, default=str)
        print(f"Equity curves saved to: {equity_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Crypto Alpha - Futures Backtest Runner")
    parser.add_argument("--symbol", type=str, help="Single symbol, e.g. BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe (default: 4h)")
    parser.add_argument("--strategy", type=str, help="Single strategy key")
    parser.add_argument("--all", action="store_true", help="Run on all coins in universe")

    args = parser.parse_args()

    if args.symbol and args.strategy:
        result, _ = run_single(args.symbol, args.timeframe, args.strategy)
        if result:
            df = load_data(args.symbol, args.timeframe)
            df = add_all_indicators(df).dropna()
            _, test_df = walk_forward_split(df, 0.7)
            signals = STRATEGIES[args.strategy]["func"](df).reindex(test_df.index)

            bt = BacktestEngine(
                test_df, initial_capital=500,
                fee=FUTURES_FEE, stop_loss_pct=FUTURES_STOP_LOSS,
                leverage=FUTURES_LEVERAGE,
            )
            bt.run(signals)
            bt.print_report()
            bt.plot_equity(f"{args.symbol} - {STRATEGIES[args.strategy]['name']} (Futures {FUTURES_LEVERAGE}x)")
    elif args.all:
        run_comparison(symbols=COIN_UNIVERSE, timeframe=args.timeframe)
    else:
        top_coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
        if args.symbol:
            top_coins = [args.symbol]
        run_comparison(symbols=top_coins, timeframe=args.timeframe)


if __name__ == "__main__":
    main()
