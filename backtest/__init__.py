"""
Backtest Engine — Vectorized
=============================
Fast backtesting engine using vectorized operations.
Supports long-only and long/short strategies with realistic cost modeling.

Usage:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(df, initial_capital=500, fee=0.001)
    results = engine.run(signals)
    engine.print_report()
"""

import numpy as np
import pandas as pd
from loguru import logger


class BacktestEngine:
    """
    Vectorized backtest engine with realistic cost modeling.

    Supports:
    - Long-only or long/short
    - Trading fees (maker/taker)
    - ATR-based or fixed stop-loss
    - Position sizing (fixed fractional)
    - Walk-forward split
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 500.0,
        fee: float = 0.001,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = None,
        stop_loss_atr: float = None,
    ):
        """
        Args:
            df: DataFrame with OHLCV + indicators (must have 'close')
            initial_capital: Starting capital in USD
            fee: Trading fee per side (0.001 = 0.1%)
            risk_per_trade: Max risk per trade as fraction of capital
            stop_loss_pct: Fixed stop-loss percentage (e.g. 0.03 = 3%)
            stop_loss_atr: Stop-loss as ATR multiplier (requires 'atr' column)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.fee = fee
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_atr = stop_loss_atr
        self.results = None

    def run(self, signals: pd.Series) -> pd.DataFrame:
        """
        Run backtest with given signals.

        Args:
            signals: Series aligned with df index.
                     1 = buy/long, -1 = sell/short, 0 = no position

        Returns:
            DataFrame with equity curve and trade details
        """
        df = self.df.copy()
        df["signal"] = signals.reindex(df.index).fillna(0).astype(int)

        # Detect position changes (entries and exits)
        df["position"] = df["signal"]
        df["trade"] = df["position"].diff().fillna(0)

        # Calculate returns
        df["market_return"] = df["close"].pct_change().fillna(0)

        # Strategy returns = position * market return
        # Shift position by 1 to avoid look-ahead bias
        # (signal at time t, enter at t+1, return measured at t+1)
        df["strategy_return_gross"] = df["position"].shift(1).fillna(0) * df["market_return"]

        # Apply trading costs on position changes
        df["cost"] = df["trade"].abs() * self.fee
        df["strategy_return"] = df["strategy_return_gross"] - df["cost"]

        # Apply stop-loss
        if self.stop_loss_pct is not None or self.stop_loss_atr is not None:
            df = self._apply_stop_loss(df)

        # Build equity curve
        df["equity"] = self.initial_capital * (1 + df["strategy_return"]).cumprod()
        df["buy_hold_equity"] = self.initial_capital * (1 + df["market_return"]).cumprod()

        # Drawdown
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

        self.results = df
        return df

    def _apply_stop_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stop-loss by zeroing out returns beyond threshold."""
        position = df["position"].shift(1).fillna(0)
        entry_price = None
        cumulative_return = 0.0

        for i in range(len(df)):
            if position.iloc[i] != 0:
                if entry_price is None:
                    entry_price = df["close"].iloc[i-1] if i > 0 else df["close"].iloc[i]

                current_return = (df["close"].iloc[i] - entry_price) / entry_price
                current_return *= position.iloc[i]  # Flip for short

                # Determine stop-loss level
                if self.stop_loss_atr is not None and "atr" in df.columns:
                    sl = self.stop_loss_atr * df["atr"].iloc[i] / entry_price
                elif self.stop_loss_pct is not None:
                    sl = self.stop_loss_pct
                else:
                    continue

                if current_return < -sl:
                    # Stop-loss hit
                    df.iloc[i, df.columns.get_loc("strategy_return")] = -sl - self.fee
                    df.iloc[i, df.columns.get_loc("position")] = 0
                    entry_price = None
            else:
                entry_price = None

        return df

    def get_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if self.results is None:
            raise ValueError("Run backtest first with .run(signals)")

        df = self.results
        returns = df["strategy_return"].dropna()
        equity = df["equity"]

        # Basic stats
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        buy_hold_return = (df["buy_hold_equity"].iloc[-1] / self.initial_capital) - 1

        # Annualized metrics
        n_periods = len(returns)
        # Determine annualization factor from data frequency
        if hasattr(df.index, 'freq') and df.index.freq:
            freq = df.index.freq
        else:
            # Estimate from median time diff
            median_diff = df.index.to_series().diff().median()
            hours = median_diff.total_seconds() / 3600
            if hours <= 1.5:
                periods_per_year = 365 * 24  # 1H
            elif hours <= 5:
                periods_per_year = 365 * 6   # 4H
            else:
                periods_per_year = 365       # 1D

        ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Drawdown stats
        max_dd = df["drawdown"].min()
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        trades = df["trade"]
        n_entries = (trades.abs() > 0).sum() // 2  # Each trade has entry + exit

        # Win/loss analysis
        trade_groups = []
        in_trade = False
        trade_return = 0

        for i in range(len(df)):
            pos = df["position"].iloc[i]
            if pos != 0 and not in_trade:
                in_trade = True
                trade_return = 0
            if in_trade:
                trade_return += df["strategy_return"].iloc[i]
            if (pos == 0 and in_trade) or (i == len(df) - 1 and in_trade):
                trade_groups.append(trade_return)
                in_trade = False
                trade_return = 0

        if trade_groups:
            wins = [t for t in trade_groups if t > 0]
            losses = [t for t in trade_groups if t <= 0]
            win_rate = len(wins) / len(trade_groups) if trade_groups else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = (
                sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
            )
            expectancy = np.mean(trade_groups)
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0

        return {
            "total_return_pct": total_return * 100,
            "buy_hold_return_pct": buy_hold_return * 100,
            "alpha_pct": (total_return - buy_hold_return) * 100,
            "ann_return_pct": ann_return * 100,
            "ann_volatility_pct": ann_vol * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_dd * 100,
            "total_trades": len(trade_groups),
            "win_rate_pct": win_rate * 100,
            "avg_win_pct": avg_win * 100,
            "avg_loss_pct": avg_loss * 100,
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy * 100,
            "total_fees_usd": (df["cost"] * self.initial_capital).sum(),
            "final_equity": equity.iloc[-1],
        }

    def print_report(self):
        """Print formatted backtest report."""
        m = self.get_metrics()

        print("\n" + "=" * 55)
        print("           BACKTEST REPORT")
        print("=" * 55)
        print(f"  Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"  Final Equity:        ${m['final_equity']:,.2f}")
        print(f"  Total Return:        {m['total_return_pct']:+.2f}%")
        print(f"  Buy & Hold Return:   {m['buy_hold_return_pct']:+.2f}%")
        print(f"  Alpha:               {m['alpha_pct']:+.2f}%")
        print("-" * 55)
        print(f"  Ann. Return:         {m['ann_return_pct']:+.2f}%")
        print(f"  Ann. Volatility:     {m['ann_volatility_pct']:.2f}%")
        print(f"  Sharpe Ratio:        {m['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:       {m['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio:        {m['calmar_ratio']:.3f}")
        print(f"  Max Drawdown:        {m['max_drawdown_pct']:.2f}%")
        print("-" * 55)
        print(f"  Total Trades:        {m['total_trades']}")
        print(f"  Win Rate:            {m['win_rate_pct']:.1f}%")
        print(f"  Avg Win:             {m['avg_win_pct']:+.3f}%")
        print(f"  Avg Loss:            {m['avg_loss_pct']:.3f}%")
        print(f"  Profit Factor:       {m['profit_factor']:.2f}")
        print(f"  Expectancy:          {m['expectancy_pct']:+.4f}%")
        print(f"  Total Fees:          ${m['total_fees_usd']:.2f}")
        print("=" * 55 + "\n")

    def plot_equity(self, title: str = "Equity Curve"):
        """Plot equity curve vs buy & hold."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed, skipping plot")
            return

        if self.results is None:
            raise ValueError("Run backtest first")

        df = self.results

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.05,
            subplot_titles=[title, "Drawdown"],
        )

        fig.add_trace(go.Scatter(
            x=df.index, y=df["equity"],
            mode="lines", name="Strategy",
            line=dict(color="blue", width=1.5),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["buy_hold_equity"],
            mode="lines", name="Buy & Hold",
            line=dict(color="gray", width=1, dash="dash"),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["drawdown"] * 100,
            fill="tozeroy", name="Drawdown",
            line=dict(color="red", width=0.5),
            fillcolor="rgba(255,0,0,0.2)",
        ), row=2, col=1)

        fig.update_layout(height=600, hovermode="x unified")
        fig.update_yaxes(title_text="Equity ($)", row=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2)
        fig.show()

        return fig


def walk_forward_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple:
    """
    Split data into train/test for walk-forward validation.

    Args:
        df: DataFrame with datetime index
        train_ratio: Fraction for training (default 0.7)

    Returns:
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(
        f"Walk-forward split: train={len(train)} rows "
        f"({train.index[0].date()} → {train.index[-1].date()}), "
        f"test={len(test)} rows "
        f"({test.index[0].date()} → {test.index[-1].date()})"
    )
    return train, test
