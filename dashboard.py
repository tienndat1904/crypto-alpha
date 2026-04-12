"""
Crypto Alpha Dashboard
=======================
Streamlit dashboard for monitoring the trading system.

Usage:
    streamlit run dashboard.py
"""

import json
import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timezone

st.set_page_config(
    page_title="Crypto Alpha Dashboard",
    page_icon="📈",
    layout="wide",
)

# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════

STATE_FILE = Path("trading/state.json")
BACKTEST_FILE = Path("backtest_results.csv")
OPTIMIZED_FILE = Path("optimized_results.csv")


@st.cache_data(ttl=30)
def load_state() -> dict:
    """Load paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": 500, "peak_capital": 500,
        "open_positions": {}, "trade_history": [],
        "consecutive_losses": 0, "total_trades": 0,
        "total_wins": 0, "total_pnl": 0.0,
    }


@st.cache_data(ttl=60)
def load_backtest_results() -> pd.DataFrame:
    if BACKTEST_FILE.exists():
        return pd.read_csv(BACKTEST_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_optimized_results() -> pd.DataFrame:
    if OPTIMIZED_FILE.exists():
        return pd.read_csv(OPTIMIZED_FILE)
    return pd.DataFrame()


# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════

st.title("Crypto Alpha Trading System")
st.caption("Real-time monitoring dashboard")

state = load_state()

# ═══════════════════════════════════════════
# TOP METRICS
# ═══════════════════════════════════════════

capital = state["capital"]
peak = state["peak_capital"]
total_pnl = state["total_pnl"]
total_trades = state["total_trades"]
total_wins = state["total_wins"]
open_count = len(state["open_positions"])

# Drawdown
total_equity = capital
for pos in state["open_positions"].values():
    total_equity += pos["size_usdt"]
drawdown = (total_equity - peak) / peak if peak > 0 else 0

win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Equity", f"${total_equity:.2f}", f"${total_pnl:+.2f}")
col2.metric("Drawdown", f"{drawdown:.1%}")
col3.metric("Open Positions", f"{open_count}/3")
col4.metric("Total Trades", total_trades)
col5.metric("Win Rate", f"{win_rate:.0f}%")
col6.metric("Consec. Losses", state["consecutive_losses"])

st.divider()

# ═══════════════════════════════════════════
# OPEN POSITIONS
# ═══════════════════════════════════════════

left, right = st.columns([1, 1])

with left:
    st.subheader("Open Positions")
    if state["open_positions"]:
        pos_data = []
        for sym, pos in state["open_positions"].items():
            pos_data.append({
                "Symbol": sym,
                "Side": pos["side"].upper(),
                "Entry": f"${pos['entry_price']:,.4f}",
                "Size": f"${pos['size_usdt']:.2f}",
                "Stop": f"${pos['stop_price']:,.4f}",
                "Opened": pos.get("opened_at", "")[:16],
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

# ═══════════════════════════════════════════
# TRADE HISTORY
# ═══════════════════════════════════════════

with right:
    st.subheader("Recent Trades")
    history = state.get("trade_history", [])
    if history:
        hist_data = []
        for t in history[-10:]:  # Last 10
            pnl_color = "+" if t["pnl_usd"] > 0 else ""
            hist_data.append({
                "Symbol": t["symbol"],
                "Side": t["side"].upper(),
                "Entry": f"${t['entry_price']:,.4f}",
                "Exit": f"${t['exit_price']:,.4f}",
                "PnL %": f"{t['pnl_pct']:+.2f}%",
                "PnL $": f"${t['pnl_usd']:+.2f}",
                "Reason": t["reason"],
            })
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
    else:
        st.info("No trade history")

st.divider()

# ═══════════════════════════════════════════
# EQUITY CURVE
# ═══════════════════════════════════════════

st.subheader("Equity Curve")

if history:
    equity_points = [{"date": state.get("created_at", "2024-01-01"), "equity": 500}]
    running_equity = 500
    for t in history:
        running_equity += t["pnl_usd"]
        equity_points.append({
            "date": t.get("closed_at", ""),
            "equity": running_equity,
        })
    # Add current
    equity_points.append({
        "date": datetime.now(timezone.utc).isoformat(),
        "equity": total_equity,
    })

    eq_df = pd.DataFrame(equity_points)
    eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
    eq_df = eq_df.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["equity"],
        mode="lines+markers",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=6),
        name="Equity",
    ))
    fig.add_hline(y=500, line_dash="dash", line_color="gray",
                  annotation_text="Initial $500")
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Equity ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Equity curve will appear after first trade")

st.divider()

# ═══════════════════════════════════════════
# BACKTEST RESULTS
# ═══════════════════════════════════════════

col_bt, col_opt = st.columns(2)

with col_bt:
    st.subheader("Backtest Results")
    bt_df = load_backtest_results()
    if not bt_df.empty:
        display_cols = ["symbol", "strategy", "sharpe_ratio", "total_return_pct",
                        "max_drawdown_pct", "win_rate_pct", "total_trades"]
        available = [c for c in display_cols if c in bt_df.columns]
        display = bt_df[available].sort_values(
            "sharpe_ratio", ascending=False
        ).head(10)
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Sharpe heatmap
        if "symbol" in bt_df.columns and "strategy" in bt_df.columns:
            pivot = bt_df.pivot_table(
                index="symbol", columns="strategy",
                values="sharpe_ratio", aggfunc="first"
            )
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(pivot.values, 2),
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                title="Sharpe Ratio by Symbol x Strategy",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Run `python -m strategies.run_backtest` first")

with col_opt:
    st.subheader("Optimized Results")
    opt_df = load_optimized_results()
    if not opt_df.empty:
        # Show available columns
        possible_cols = ["symbol", "strategy", "avg_test_sharpe", "std_test_sharpe",
                         "avg_test_return", "avg_test_dd", "positive_folds", "n_folds",
                         "test_sharpe", "test_return", "test_dd"]
        available = [c for c in possible_cols if c in opt_df.columns]
        if available:
            display = opt_df[available].sort_values(
                available[2] if len(available) > 2 else available[0], ascending=False
            )
            st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m strategies.run_optimized` first")

st.divider()

# ═══════════════════════════════════════════
# PNL BREAKDOWN
# ═══════════════════════════════════════════

if history and len(history) >= 2:
    st.subheader("PnL Analysis")

    pnl_col1, pnl_col2 = st.columns(2)

    with pnl_col1:
        # PnL per trade bar chart
        pnl_data = pd.DataFrame(history)
        fig_pnl = go.Figure()
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_data["pnl_usd"]]
        fig_pnl.add_trace(go.Bar(
            x=list(range(1, len(pnl_data) + 1)),
            y=pnl_data["pnl_usd"],
            marker_color=colors,
            name="PnL per Trade",
        ))
        fig_pnl.update_layout(
            title="PnL per Trade ($)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Trade #",
            yaxis_title="PnL ($)",
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    with pnl_col2:
        # PnL by symbol
        pnl_by_sym = pnl_data.groupby("symbol")["pnl_usd"].sum().sort_values()
        fig_sym = go.Figure()
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_by_sym.values]
        fig_sym.add_trace(go.Bar(
            x=pnl_by_sym.index,
            y=pnl_by_sym.values,
            marker_color=colors,
        ))
        fig_sym.update_layout(
            title="Total PnL by Symbol ($)",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="PnL ($)",
        )
        st.plotly_chart(fig_sym, use_container_width=True)

# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.divider()
st.caption(
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
    f"Auto-refresh: 30s"
)

# Auto-refresh every 30 seconds
st.markdown(
    """<meta http-equiv="refresh" content="30">""",
    unsafe_allow_html=True,
)
