"""
Crypto Alpha Dashboard
=======================
Full-featured Streamlit dashboard with Binance dark theme,
bilingual support (Vietnamese / English), and tabbed navigation.

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
from datetime import datetime, timezone, timedelta

try:
    import ccxt
except ImportError:
    ccxt = None

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Crypto Alpha Dashboard",
    page_icon="📈",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# BINANCE DARK THEME CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #181A20;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #181A20;
    border-bottom: 1px solid #2B3139;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E2329;
    border-right: 1px solid #2B3139;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #EAECEF;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #1E2329;
    border: 1px solid #2B3139;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label {
    color: #848E9C !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #EAECEF !important;
}

/* Tabs */
div[data-testid="stTabs"] {
    background-color: #1E2329;
    border-radius: 8px;
}
button[data-baseweb="tab"] {
    color: #848E9C !important;
    background-color: #1E2329 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #EAECEF !important;
    border-bottom: 2px solid #FCD535 !important;
}

/* Tables / dataframes */
div[data-testid="stDataFrame"] {
    background-color: #1E2329;
}
div[data-testid="stDataFrame"] table thead tr {
    background-color: #2B3139;
}

/* Buttons */
button[kind="primary"], .stButton > button {
    background-color: #FCD535 !important;
    color: #181A20 !important;
    border: none !important;
    font-weight: 600;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background-color: #e5c230 !important;
}

/* General text */
.stMarkdown, .stText, p, span, label {
    color: #EAECEF;
}

/* Divider */
hr {
    border-color: #2B3139;
}

/* Info / warning boxes */
div[data-testid="stAlert"] {
    background-color: #1E2329;
    border: 1px solid #2B3139;
    color: #EAECEF;
}

/* Select boxes */
div[data-baseweb="select"] {
    background-color: #1E2329;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# COLOR CONSTANTS
# ═══════════════════════════════════════════════════════════════

BINANCE_GREEN = "#0ECB81"
BINANCE_RED = "#F6465D"
BINANCE_YELLOW = "#FCD535"

# ═══════════════════════════════════════════════════════════════
# PLOTLY THEME WRAPPER
# ═══════════════════════════════════════════════════════════════


def apply_binance_theme(fig: go.Figure) -> go.Figure:
    """Apply Binance dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#181A20",
        plot_bgcolor="#1E2329",
        font=dict(color="#EAECEF"),
        xaxis=dict(gridcolor="#2B3139", zerolinecolor="#2B3139"),
        yaxis=dict(gridcolor="#2B3139", zerolinecolor="#2B3139"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# LANGUAGE DICTIONARY
# ═══════════════════════════════════════════════════════════════

LANG = {
    "vi": {
        "title": "Crypto Alpha Trading System",
        "subtitle": "Bảng điều khiển giám sát thời gian thực",
        "tab_market": "Thị trường",
        "tab_trading": "Giao dịch",
        "tab_backtest": "Phân tích Backtest",
        "tab_performance": "Hiệu suất",
        "market_watch": "Giá thị trường",
        "overview_24h": "Tổng quan 24h",
        "candlestick_chart": "Biểu đồ nến",
        "select_coin": "Chọn coin",
        "period": "Khung thời gian",
        "sparklines": "Biểu đồ mini",
        "equity": "Tài sản ròng",
        "drawdown": "Drawdown",
        "open_positions": "Vị thế đang mở",
        "no_open_positions": "Không có vị thế đang mở",
        "recent_trades": "Giao dịch gần đây",
        "no_trades": "Chưa có lịch sử giao dịch",
        "trade_history": "Lịch sử giao dịch",
        "equity_curve": "Đường cong Equity",
        "equity_after_trade": "Equity sau giao dịch",
        "total_trades": "Tổng giao dịch",
        "win_rate": "Tỉ lệ thắng",
        "consec_losses": "Thua liên tiếp",
        "cash": "Tiền mặt",
        "peak": "Đỉnh",
        "backtest_analysis": "Phân tích Backtest",
        "no_backtest": "Chưa có dữ liệu backtest. Chạy backtest trước.",
        "overview": "Tổng quan",
        "equity_curves": "Đường cong Equity",
        "rolling_metrics": "Chỉ số cuộn",
        "strategy_compare": "So sánh chiến lược",
        "detail_tables": "Bảng chi tiết",
        "by_strategy": "Theo chiến lược",
        "by_coin": "Theo coin",
        "signal_log": "Nhật ký tín hiệu",
        "no_signal_log": "Chưa có nhật ký tín hiệu",
        "performance": "Hiệu suất",
        "strategy_stats": "Thống kê chiến lược",
        "coin_stats": "Thống kê theo coin",
        "exit_reasons": "Lý do thoát lệnh",
        "pnl_per_trade": "Lãi/Lỗ mỗi giao dịch",
        "pnl_by_symbol": "Lãi/Lỗ theo Symbol",
        "pnl_by_strategy": "Lãi/Lỗ theo chiến lược",
        "pnl_analysis": "Phân tích Lãi/Lỗ",
        "total_return": "Tổng lợi nhuận",
        "filter_action": "Lọc hành động",
        "filter_symbol": "Lọc symbol",
        "filter_strategy": "Lọc chiến lược",
        "all": "Tất cả",
        "last_updated": "Cập nhật lần cuối",
        "auto_refresh": "Tự động làm mới",
        "train_period": "Giai đoạn huấn luyện",
        "test_period": "Giai đoạn kiểm tra",
        "initial_capital": "Vốn ban đầu",
        "spot_trading": "Giao dịch Spot",
        "futures_trading": "Giao dịch Futures",
        "futures_status": "Trạng thái Futures",
        "leverage": "Đòn bẩy",
        "margin_type": "Loại margin",
        "margin_used": "Margin đã dùng",
        "notional": "Giá trị danh nghĩa",
        "liq_price": "Giá thanh lý",
        "roe": "ROE",
        "no_futures_data": "Chưa có dữ liệu giao dịch Futures",
        "unrealized_pnl": "Lãi/Lỗ chưa chốt",
        "sharpe_heatmap": "Heatmap tỉ lệ Sharpe",
        "sharpe_by_sym_strat": "Sharpe theo Symbol x Chiến lược",
        "optimized_results": "Kết quả tối ưu hóa",
        "backtest_results": "Kết quả Backtest",
        "pnl_long_short": "Lãi/Lỗ Long vs Short",
        "positions": "Vị thế",
        "side": "Hướng",
        "entry": "Giá vào",
        "exit": "Giá ra",
        "size": "Kích thước",
        "stop": "Dừng lỗ",
        "reason": "Lý do",
        "opened_at": "Thời gian mở",
        "symbol": "Symbol",
        "no_data": "Không có dữ liệu",
        "price": "Giá",
        "change_24h": "Thay đổi 24h",
        "volume_24h": "Khối lượng 24h",
    },
    "en": {
        "title": "Crypto Alpha Trading System",
        "subtitle": "Real-time monitoring dashboard",
        "tab_market": "Market",
        "tab_trading": "Trading",
        "tab_backtest": "Backtest Analysis",
        "tab_performance": "Performance",
        "market_watch": "Market Watch",
        "overview_24h": "24h Overview",
        "candlestick_chart": "Candlestick Chart",
        "select_coin": "Select coin",
        "period": "Period",
        "sparklines": "Sparklines",
        "equity": "Net Equity",
        "drawdown": "Drawdown",
        "open_positions": "Open Positions",
        "no_open_positions": "No open positions",
        "recent_trades": "Recent Trades",
        "no_trades": "No trade history",
        "trade_history": "Trade History",
        "equity_curve": "Equity Curve",
        "equity_after_trade": "Equity after trade",
        "total_trades": "Total Trades",
        "win_rate": "Win Rate",
        "consec_losses": "Consec. Losses",
        "cash": "Cash",
        "peak": "Peak",
        "backtest_analysis": "Backtest Analysis",
        "no_backtest": "No backtest data. Run backtest first.",
        "overview": "Overview",
        "equity_curves": "Equity Curves",
        "rolling_metrics": "Rolling Metrics",
        "strategy_compare": "Strategy Comparison",
        "detail_tables": "Detail Tables",
        "by_strategy": "By Strategy",
        "by_coin": "By Coin",
        "signal_log": "Signal Log",
        "no_signal_log": "No signal log available",
        "performance": "Performance",
        "strategy_stats": "Strategy Stats",
        "coin_stats": "Coin Stats",
        "exit_reasons": "Exit Reasons",
        "pnl_per_trade": "PnL per Trade",
        "pnl_by_symbol": "PnL by Symbol",
        "pnl_by_strategy": "PnL by Strategy",
        "pnl_analysis": "PnL Analysis",
        "total_return": "Total Return",
        "filter_action": "Filter Action",
        "filter_symbol": "Filter Symbol",
        "filter_strategy": "Filter Strategy",
        "all": "All",
        "last_updated": "Last updated",
        "auto_refresh": "Auto-refresh",
        "train_period": "Train Period",
        "test_period": "Test Period",
        "initial_capital": "Initial Capital",
        "spot_trading": "Spot Trading",
        "futures_trading": "Futures Trading",
        "futures_status": "Futures Status",
        "leverage": "Leverage",
        "margin_type": "Margin Type",
        "margin_used": "Margin Used",
        "notional": "Notional",
        "liq_price": "Liq. Price",
        "roe": "ROE",
        "no_futures_data": "No futures trading data available",
        "unrealized_pnl": "Unrealized PnL",
        "sharpe_heatmap": "Sharpe Ratio Heatmap",
        "sharpe_by_sym_strat": "Sharpe by Symbol x Strategy",
        "optimized_results": "Optimized Results",
        "backtest_results": "Backtest Results",
        "pnl_long_short": "PnL Long vs Short",
        "positions": "Positions",
        "side": "Side",
        "entry": "Entry",
        "exit": "Exit",
        "size": "Size",
        "stop": "Stop",
        "reason": "Reason",
        "opened_at": "Opened At",
        "symbol": "Symbol",
        "no_data": "No data",
        "price": "Price",
        "change_24h": "24h Change",
        "volume_24h": "24h Volume",
    },
}

# ═══════════════════════════════════════════════════════════════
# TRANSLATION FUNCTION
# ═══════════════════════════════════════════════════════════════

if "lang" not in st.session_state:
    st.session_state.lang = "vi"


def t(key: str) -> str:
    """Get translated string for current language."""
    lang = st.session_state.lang
    return LANG.get(lang, LANG["en"]).get(key, key)


# ═══════════════════════════════════════════════════════════════
# DATA FILE PATHS
# ═══════════════════════════════════════════════════════════════

STATE_FILE = Path("trading/state.json")
FUTURES_STATE_FILE = Path("trading/futures_state.json")
BACKTEST_FILE = Path("backtest_results.csv")
OPTIMIZED_FILE = Path("optimized_results.csv")
EQUITY_JSON_FILE = Path("backtest_equity.json")
SIGNAL_LOG_FILE = Path("logs/signal_history.jsonl")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def load_state() -> dict:
    """Load spot paper trading state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital": 500, "peak_capital": 500,
        "open_positions": {}, "trade_history": [],
        "consecutive_losses": 0, "total_trades": 0,
        "total_wins": 0, "total_pnl": 0.0,
    }


@st.cache_data(ttl=300)
def load_futures_state() -> dict | None:
    """Load futures trading state. Returns None if not exists."""
    if FUTURES_STATE_FILE.exists():
        with open(FUTURES_STATE_FILE) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_backtest_results() -> pd.DataFrame:
    """Load backtest results CSV."""
    if BACKTEST_FILE.exists():
        return pd.read_csv(BACKTEST_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_optimized_results() -> pd.DataFrame:
    """Load optimized / walk-forward results CSV."""
    if OPTIMIZED_FILE.exists():
        return pd.read_csv(OPTIMIZED_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_equity_curves() -> dict:
    """Load backtest equity curves from JSON."""
    if EQUITY_JSON_FILE.exists():
        with open(EQUITY_JSON_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_signal_log() -> pd.DataFrame:
    """Load signal history JSONL."""
    if SIGNAL_LOG_FILE.exists():
        lines = []
        with open(SIGNAL_LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if lines:
            return pd.DataFrame(lines)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_tickers():
    """Fetch market tickers from Binance via ccxt."""
    if ccxt is None:
        return {}
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        tickers = exchange.fetch_tickers()
        return tickers
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Fetch OHLCV data from Binance."""
    if ccxt is None:
        return pd.DataFrame()
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# HEADER + LANGUAGE SELECTOR
# ═══════════════════════════════════════════════════════════════

header_left, header_right = st.columns([4, 1])

with header_left:
    st.title(t("title"))
    st.caption(t("subtitle"))

with header_right:
    lang_options = {"Tiếng Việt": "vi", "English": "en"}
    selected_lang_label = st.selectbox(
        "🌐",
        options=list(lang_options.keys()),
        index=0 if st.session_state.lang == "vi" else 1,
        label_visibility="collapsed",
    )
    new_lang = lang_options[selected_lang_label]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════
# WATCHED COINS
# ═══════════════════════════════════════════════════════════════

WATCHED_COINS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
    "DOT/USDT", "MATIC/USDT", "LINK/USDT", "UNI/USDT",
]

# ═══════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ═══════════════════════════════════════════════════════════════

tab_market, tab_trading, tab_backtest, tab_performance = st.tabs([
    t("tab_market"),
    t("tab_trading"),
    t("tab_backtest"),
    t("tab_performance"),
])

# ═══════════════════════════════════════════════════════════════
# MARKET TAB
# ═══════════════════════════════════════════════════════════════

with tab_market:
    st.subheader(t("market_watch"))

    # --- Ticker cards ---
    tickers_data = fetch_tickers()

    if tickers_data:
        st.markdown(f"#### {t('overview_24h')}")
        ticker_cols = st.columns(4)
        for idx, coin in enumerate(WATCHED_COINS):
            tk = tickers_data.get(coin, None)
            if tk is None:
                continue
            col = ticker_cols[idx % 4]
            last_price = tk.get("last", 0)
            change_pct = tk.get("percentage", 0) or 0
            vol_24h = tk.get("quoteVolume", 0) or 0
            color = BINANCE_GREEN if change_pct >= 0 else BINANCE_RED
            sign = "+" if change_pct >= 0 else ""
            with col:
                st.markdown(
                    f"""
                    <div style="background:#1E2329; border:1px solid #2B3139;
                                border-radius:8px; padding:12px; margin-bottom:8px;">
                        <div style="color:#848E9C; font-size:12px;">{coin}</div>
                        <div style="color:#EAECEF; font-size:20px; font-weight:700;">
                            ${last_price:,.2f}
                        </div>
                        <div style="color:{color}; font-size:14px;">
                            {sign}{change_pct:.2f}%
                        </div>
                        <div style="color:#848E9C; font-size:11px;">
                            {t('volume_24h')}: ${vol_24h:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # --- Candlestick chart ---
        st.markdown(f"#### {t('candlestick_chart')}")
        chart_col1, chart_col2 = st.columns([2, 1])
        with chart_col1:
            selected_coin = st.selectbox(
                t("select_coin"),
                options=WATCHED_COINS,
                index=0,
                key="candle_coin",
            )
        with chart_col2:
            selected_period = st.selectbox(
                t("period"),
                options=["1h", "4h", "1d"],
                index=0,
                key="candle_period",
            )

        ohlcv_df = fetch_ohlcv(selected_coin, selected_period, limit=100)
        if not ohlcv_df.empty:
            fig_candle = go.Figure(data=[
                go.Candlestick(
                    x=ohlcv_df["timestamp"],
                    open=ohlcv_df["open"],
                    high=ohlcv_df["high"],
                    low=ohlcv_df["low"],
                    close=ohlcv_df["close"],
                    increasing_line_color=BINANCE_GREEN,
                    decreasing_line_color=BINANCE_RED,
                    increasing_fillcolor=BINANCE_GREEN,
                    decreasing_fillcolor=BINANCE_RED,
                    name=selected_coin,
                )
            ])
            fig_candle.update_layout(
                title=f"{selected_coin} - {selected_period}",
                height=450,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_rangeslider_visible=False,
                yaxis_title=t("price"),
                hovermode="x unified",
            )
            apply_binance_theme(fig_candle)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.info(t("no_data"))

        st.divider()

        # --- Sparklines ---
        st.markdown(f"#### {t('sparklines')}")
        spark_cols = st.columns(4)
        for idx, coin in enumerate(WATCHED_COINS):
            col = spark_cols[idx % 4]
            spark_df = fetch_ohlcv(coin, "1h", limit=48)
            if spark_df.empty:
                continue
            with col:
                fig_spark = go.Figure()
                close_vals = spark_df["close"]
                line_color = BINANCE_GREEN if close_vals.iloc[-1] >= close_vals.iloc[0] else BINANCE_RED
                fig_spark.add_trace(go.Scatter(
                    x=spark_df["timestamp"],
                    y=close_vals,
                    mode="lines",
                    line=dict(color=line_color, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig_spark.update_layout(
                    height=80,
                    margin=dict(l=0, r=0, t=18, b=0),
                    title=dict(text=coin.split("/")[0], font=dict(size=10, color="#848E9C")),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                apply_binance_theme(fig_spark)
                st.plotly_chart(fig_spark, use_container_width=True, key=f"spark_{coin}")
    else:
        st.info(t("no_data") + " (ccxt not available or API error)")


# ═══════════════════════════════════════════════════════════════
# TRADING TAB
# ═══════════════════════════════════════════════════════════════

with tab_trading:
    trading_spot_tab, trading_futures_tab = st.tabs([
        t("spot_trading"),
        t("futures_trading"),
    ])

    # ─── SPOT TRADING ─────────────────────────────────────────
    with trading_spot_tab:
        state = load_state()
        capital = state["capital"]
        peak = state["peak_capital"]
        total_pnl = state["total_pnl"]
        total_trades_count = state["total_trades"]
        total_wins = state["total_wins"]
        open_pos = state["open_positions"]
        open_count = len(open_pos)
        history = state.get("trade_history", [])

        # Equity calculation
        total_equity = capital
        for pos in open_pos.values():
            total_equity += pos.get("size_usdt", 0)
        drawdown = (total_equity - peak) / peak if peak > 0 else 0
        win_rate = (total_wins / total_trades_count * 100) if total_trades_count > 0 else 0

        # --- Metrics row ---
        st.markdown(f"#### {t('spot_trading')}")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(t("equity"), f"${total_equity:.2f}", f"${total_pnl:+.2f}")
        m2.metric(t("drawdown"), f"{drawdown:.1%}")
        m3.metric(t("open_positions"), f"{open_count}/3")
        m4.metric(t("total_trades"), total_trades_count)
        m5.metric(t("win_rate"), f"{win_rate:.0f}%")
        m6.metric(t("consec_losses"), state["consecutive_losses"])

        st.divider()

        # --- Open Positions ---
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(f"##### {t('open_positions')}")
            if open_pos:
                pos_rows = []
                for sym, pos in open_pos.items():
                    row = {
                        t("symbol"): sym,
                        t("side"): pos["side"].upper(),
                        t("entry"): f"${pos['entry_price']:,.4f}",
                        t("size"): f"${pos.get('size_usdt', 0):.2f}",
                        t("stop"): f"${pos['stop_price']:,.4f}",
                        "TP1": f"${pos['tp1_price']:,.4f}" if "tp1_price" in pos else "-",
                        t("opened_at"): pos.get("opened_at", "")[:16],
                    }
                    pos_rows.append(row)
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
            else:
                st.info(t("no_open_positions"))

        # --- Recent Trades ---
        with right_col:
            st.markdown(f"##### {t('recent_trades')}")
            if history:
                hist_rows = []
                for trade in history[-10:]:
                    hist_rows.append({
                        t("symbol"): trade["symbol"],
                        t("side"): trade["side"].upper(),
                        t("entry"): f"${trade['entry_price']:,.4f}",
                        t("exit"): f"${trade['exit_price']:,.4f}",
                        "PnL %": f"{trade['pnl_pct']:+.2f}%",
                        "PnL $": f"${trade['pnl_usd']:+.2f}",
                        t("reason"): trade["reason"],
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
            else:
                st.info(t("no_trades"))

        st.divider()

        # --- Equity Curve ---
        st.markdown(f"##### {t('equity_curve')}")
        if history:
            equity_points = [{"date": state.get("created_at", "2024-01-01"), "equity": 500}]
            running_equity = 500
            for trade in history:
                running_equity += trade["pnl_usd"]
                equity_points.append({
                    "date": trade.get("closed_at", ""),
                    "equity": running_equity,
                })
            equity_points.append({
                "date": datetime.now(timezone.utc).isoformat(),
                "equity": total_equity,
            })

            eq_df = pd.DataFrame(equity_points)
            eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
            eq_df = eq_df.dropna()

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["equity"],
                mode="lines+markers",
                line=dict(color=BINANCE_YELLOW, width=2),
                marker=dict(size=5, color=BINANCE_YELLOW),
                name=t("equity"),
                fill="tozeroy",
                fillcolor="rgba(252,213,53,0.08)",
            ))
            fig_eq.add_hline(
                y=500, line_dash="dash", line_color="#848E9C",
                annotation_text=f"{t('initial_capital')} $500",
                annotation_font_color="#848E9C",
            )
            fig_eq.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title=t("equity") + " ($)",
                hovermode="x unified",
            )
            apply_binance_theme(fig_eq)
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info(t("equity_after_trade"))

    # ─── FUTURES TRADING ──────────────────────────────────────
    with trading_futures_tab:
        futures_state = load_futures_state()

        if futures_state is not None:
            f_capital = futures_state.get("capital", 0)
            f_peak = futures_state.get("peak_capital", 0)
            f_total_pnl = futures_state.get("total_pnl", 0)
            f_total_trades = futures_state.get("total_trades", 0)
            f_total_wins = futures_state.get("total_wins", 0)
            f_open_pos = futures_state.get("open_positions", {})
            f_open_count = len(f_open_pos)
            f_history = futures_state.get("trade_history", [])
            f_leverage = futures_state.get("leverage", 1)
            f_margin_used = futures_state.get("margin_used", 0)
            f_consec_losses = futures_state.get("consecutive_losses", 0)

            f_total_equity = f_capital
            for pos in f_open_pos.values():
                f_total_equity += pos.get("size_usdt", 0)
            f_drawdown = (f_total_equity - f_peak) / f_peak if f_peak > 0 else 0
            f_win_rate = (f_total_wins / f_total_trades * 100) if f_total_trades > 0 else 0

            # --- Metrics ---
            st.markdown(f"#### {t('futures_trading')}")
            fm1, fm2, fm3, fm4, fm5, fm6 = st.columns(6)
            fm1.metric(t("equity"), f"${f_total_equity:.2f}", f"${f_total_pnl:+.2f}")
            fm2.metric(t("leverage"), f"{f_leverage}x")
            fm3.metric(t("margin_used"), f"${f_margin_used:.2f}")
            fm4.metric(t("open_positions"), f"{f_open_count}")
            fm5.metric(t("win_rate"), f"{f_win_rate:.0f}%")
            fm6.metric(t("drawdown"), f"{f_drawdown:.1%}")

            st.divider()

            # --- Open Positions ---
            st.markdown(f"##### {t('open_positions')}")
            if f_open_pos:
                f_pos_rows = []
                for sym, pos in f_open_pos.items():
                    entry_p = pos.get("entry_price", 0)
                    size_u = pos.get("size_usdt", 0)
                    lev = pos.get("leverage", f_leverage)
                    notional_val = size_u * lev
                    margin_val = size_u
                    liq_p = pos.get("liq_price", 0)
                    tp1_val = f"${pos['tp1_price']:,.4f}" if "tp1_price" in pos else "-"
                    tp2_val = f"${pos['tp2_price']:,.4f}" if "tp2_price" in pos else "-"
                    unrealized = pos.get("unrealized_pnl", 0)

                    f_pos_rows.append({
                        t("symbol"): sym,
                        t("side"): pos["side"].upper(),
                        t("entry"): f"${entry_p:,.4f}",
                        t("leverage"): f"{lev}x",
                        t("notional"): f"${notional_val:,.2f}",
                        t("margin_used"): f"${margin_val:.2f}",
                        t("liq_price"): f"${liq_p:,.4f}" if liq_p else "-",
                        "TP1": tp1_val,
                        "TP2": tp2_val,
                        t("stop"): f"${pos.get('stop_price', 0):,.4f}",
                        t("unrealized_pnl"): f"${unrealized:+.2f}",
                    })
                st.dataframe(pd.DataFrame(f_pos_rows), use_container_width=True, hide_index=True)
            else:
                st.info(t("no_open_positions"))

            st.divider()

            # --- Futures Trade History ---
            st.markdown(f"##### {t('trade_history')}")
            if f_history:
                f_hist_rows = []
                for trade in f_history[-15:]:
                    roe_val = trade.get("roe", 0)
                    f_hist_rows.append({
                        t("symbol"): trade["symbol"],
                        t("side"): trade["side"].upper(),
                        t("entry"): f"${trade['entry_price']:,.4f}",
                        t("exit"): f"${trade['exit_price']:,.4f}",
                        "PnL $": f"${trade['pnl_usd']:+.2f}",
                        "PnL %": f"{trade['pnl_pct']:+.2f}%",
                        t("roe"): f"{roe_val:+.2f}%",
                        t("reason"): trade.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(f_hist_rows), use_container_width=True, hide_index=True)
            else:
                st.info(t("no_trades"))

            st.divider()

            # --- Futures Equity Curve ---
            st.markdown(f"##### {t('equity_curve')}")
            if f_history:
                f_eq_points = [{
                    "date": futures_state.get("created_at", "2024-01-01"),
                    "equity": futures_state.get("initial_capital", 500),
                }]
                f_running = f_eq_points[0]["equity"]
                for trade in f_history:
                    f_running += trade["pnl_usd"]
                    f_eq_points.append({
                        "date": trade.get("closed_at", ""),
                        "equity": f_running,
                    })
                f_eq_points.append({
                    "date": datetime.now(timezone.utc).isoformat(),
                    "equity": f_total_equity,
                })

                f_eq_df = pd.DataFrame(f_eq_points)
                f_eq_df["date"] = pd.to_datetime(f_eq_df["date"], errors="coerce")
                f_eq_df = f_eq_df.dropna()

                fig_feq = go.Figure()
                fig_feq.add_trace(go.Scatter(
                    x=f_eq_df["date"], y=f_eq_df["equity"],
                    mode="lines+markers",
                    line=dict(color=BINANCE_YELLOW, width=2),
                    marker=dict(size=5, color=BINANCE_YELLOW),
                    name=t("equity"),
                    fill="tozeroy",
                    fillcolor="rgba(252,213,53,0.08)",
                ))
                fig_feq.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("equity") + " ($)",
                    hovermode="x unified",
                )
                apply_binance_theme(fig_feq)
                st.plotly_chart(fig_feq, use_container_width=True)
            else:
                st.info(t("equity_after_trade"))
        else:
            st.info(t("no_futures_data"))


# ═══════════════════════════════════════════════════════════════
# BACKTEST TAB
# ═══════════════════════════════════════════════════════════════

with tab_backtest:
    bt_df = load_backtest_results()
    opt_df = load_optimized_results()
    eq_curves = load_equity_curves()

    if bt_df.empty and opt_df.empty:
        st.info(t("no_backtest"))
    else:
        bt_sub_overview, bt_sub_equity, bt_sub_rolling, bt_sub_compare, bt_sub_detail = st.tabs([
            t("overview"),
            t("equity_curves"),
            t("rolling_metrics"),
            t("strategy_compare"),
            t("detail_tables"),
        ])

        # ─── Overview ─────────────────────────────────────────
        with bt_sub_overview:
            st.markdown(f"#### {t('backtest_results')}")
            if not bt_df.empty:
                display_cols = [
                    "symbol", "strategy", "sharpe_ratio", "total_return_pct",
                    "max_drawdown_pct", "win_rate_pct", "total_trades",
                ]
                available = [c for c in display_cols if c in bt_df.columns]
                if available:
                    sorted_df = bt_df[available].sort_values(
                        "sharpe_ratio", ascending=False
                    ) if "sharpe_ratio" in available else bt_df[available]
                    st.dataframe(sorted_df, use_container_width=True, hide_index=True)

                # Sharpe Heatmap
                if "symbol" in bt_df.columns and "strategy" in bt_df.columns and "sharpe_ratio" in bt_df.columns:
                    st.markdown(f"##### {t('sharpe_heatmap')}")
                    pivot = bt_df.pivot_table(
                        index="symbol", columns="strategy",
                        values="sharpe_ratio", aggfunc="first",
                    )
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale=[
                            [0, BINANCE_RED],
                            [0.5, "#2B3139"],
                            [1, BINANCE_GREEN],
                        ],
                        zmid=0,
                        text=np.round(pivot.values, 2),
                        texttemplate="%{text}",
                        textfont=dict(color="#EAECEF"),
                        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                    ))
                    fig_heat.update_layout(
                        title=t("sharpe_by_sym_strat"),
                        height=max(300, len(pivot.index) * 35),
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    apply_binance_theme(fig_heat)
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info(t("no_backtest"))

        # ─── Equity Curves ────────────────────────────────────
        with bt_sub_equity:
            st.markdown(f"#### {t('equity_curves')}")
            if eq_curves:
                # Each key is a strategy/symbol combo
                fig_eq_bt = go.Figure()
                color_palette = [
                    BINANCE_YELLOW, BINANCE_GREEN, "#3861FB", "#E040FB",
                    "#FF6D00", "#00E5FF", "#76FF03", "#FF1744",
                    "#FFAB00", "#D500F9", "#00B0FF", "#64DD17",
                ]
                for i, (curve_name, curve_data) in enumerate(eq_curves.items()):
                    if isinstance(curve_data, dict):
                        dates = list(curve_data.keys())
                        values = list(curve_data.values())
                    elif isinstance(curve_data, list):
                        dates = list(range(len(curve_data)))
                        values = curve_data
                    else:
                        continue

                    fig_eq_bt.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode="lines",
                        name=curve_name,
                        line=dict(
                            color=color_palette[i % len(color_palette)],
                            width=1.5,
                        ),
                    ))

                fig_eq_bt.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("equity"),
                    hovermode="x unified",
                    legend=dict(
                        bgcolor="rgba(30,35,41,0.8)",
                        font=dict(color="#EAECEF"),
                    ),
                )
                apply_binance_theme(fig_eq_bt)
                st.plotly_chart(fig_eq_bt, use_container_width=True)
            else:
                st.info(t("no_data"))

        # ─── Rolling Metrics ─────────────────────────────────
        with bt_sub_rolling:
            st.markdown(f"#### {t('rolling_metrics')}")
            # Rolling metrics from backtest results (if rolling columns exist)
            rolling_cols = [c for c in bt_df.columns if "rolling" in c.lower()] if not bt_df.empty else []
            if rolling_cols:
                for rc in rolling_cols:
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(
                        y=bt_df[rc],
                        mode="lines",
                        line=dict(color=BINANCE_YELLOW, width=1.5),
                        name=rc,
                    ))
                    fig_roll.update_layout(
                        title=rc,
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    apply_binance_theme(fig_roll)
                    st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.info(t("no_data"))

        # ─── Strategy Comparison ──────────────────────────────
        with bt_sub_compare:
            st.markdown(f"#### {t('strategy_compare')}")
            if not bt_df.empty and "strategy" in bt_df.columns:
                # Average Sharpe by strategy
                if "sharpe_ratio" in bt_df.columns:
                    avg_sharpe = bt_df.groupby("strategy")["sharpe_ratio"].mean().sort_values(ascending=False)
                    fig_sharpe_bar = go.Figure()
                    bar_colors = [BINANCE_GREEN if v > 0 else BINANCE_RED for v in avg_sharpe.values]
                    fig_sharpe_bar.add_trace(go.Bar(
                        x=avg_sharpe.index,
                        y=avg_sharpe.values,
                        marker_color=bar_colors,
                        text=[f"{v:.2f}" for v in avg_sharpe.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_sharpe_bar.update_layout(
                        title="Avg Sharpe Ratio",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="Sharpe",
                    )
                    apply_binance_theme(fig_sharpe_bar)
                    st.plotly_chart(fig_sharpe_bar, use_container_width=True)

                # Average Return by strategy
                if "total_return_pct" in bt_df.columns:
                    avg_ret = bt_df.groupby("strategy")["total_return_pct"].mean().sort_values(ascending=False)
                    fig_ret_bar = go.Figure()
                    bar_colors = [BINANCE_GREEN if v > 0 else BINANCE_RED for v in avg_ret.values]
                    fig_ret_bar.add_trace(go.Bar(
                        x=avg_ret.index,
                        y=avg_ret.values,
                        marker_color=bar_colors,
                        text=[f"{v:.1f}%" for v in avg_ret.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_ret_bar.update_layout(
                        title=f"Avg {t('total_return')} (%)",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="%",
                    )
                    apply_binance_theme(fig_ret_bar)
                    st.plotly_chart(fig_ret_bar, use_container_width=True)

                # Win Rate by strategy
                if "win_rate_pct" in bt_df.columns:
                    avg_wr = bt_df.groupby("strategy")["win_rate_pct"].mean().sort_values(ascending=False)
                    fig_wr_bar = go.Figure()
                    fig_wr_bar.add_trace(go.Bar(
                        x=avg_wr.index,
                        y=avg_wr.values,
                        marker_color=BINANCE_YELLOW,
                        text=[f"{v:.1f}%" for v in avg_wr.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_wr_bar.update_layout(
                        title=f"Avg {t('win_rate')} (%)",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="%",
                    )
                    apply_binance_theme(fig_wr_bar)
                    st.plotly_chart(fig_wr_bar, use_container_width=True)

                # Max Drawdown by strategy
                if "max_drawdown_pct" in bt_df.columns:
                    avg_dd = bt_df.groupby("strategy")["max_drawdown_pct"].mean().sort_values()
                    fig_dd_bar = go.Figure()
                    fig_dd_bar.add_trace(go.Bar(
                        x=avg_dd.index,
                        y=avg_dd.values,
                        marker_color=BINANCE_RED,
                        text=[f"{v:.1f}%" for v in avg_dd.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_dd_bar.update_layout(
                        title=f"Avg {t('drawdown')} (%)",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="%",
                    )
                    apply_binance_theme(fig_dd_bar)
                    st.plotly_chart(fig_dd_bar, use_container_width=True)
            else:
                st.info(t("no_data"))

        # ─── Detail Tables ────────────────────────────────────
        with bt_sub_detail:
            st.markdown(f"#### {t('detail_tables')}")

            if not opt_df.empty:
                st.markdown(f"##### {t('optimized_results')}")
                possible_cols = [
                    "symbol", "strategy", "avg_test_sharpe", "std_test_sharpe",
                    "avg_test_return", "avg_test_dd", "positive_folds", "n_folds",
                    "test_sharpe", "test_return", "test_dd",
                ]
                available = [c for c in possible_cols if c in opt_df.columns]
                if available:
                    sort_col = "avg_test_sharpe" if "avg_test_sharpe" in available else (
                        "test_sharpe" if "test_sharpe" in available else available[0]
                    )
                    display = opt_df[available].sort_values(sort_col, ascending=False)
                    st.dataframe(display, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(opt_df, use_container_width=True, hide_index=True)

            if not bt_df.empty:
                st.markdown(f"##### {t('backtest_results')} ({t('all')})")
                st.dataframe(bt_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE TAB
# ═══════════════════════════════════════════════════════════════

with tab_performance:
    perf_spot_tab, perf_futures_tab = st.tabs([
        t("spot_trading"),
        t("futures_trading"),
    ])

    # ─── SPOT PERFORMANCE ─────────────────────────────────────
    with perf_spot_tab:
        state = load_state()
        history = state.get("trade_history", [])

        if history and len(history) >= 1:
            pnl_data = pd.DataFrame(history)

            st.markdown(f"#### {t('pnl_analysis')}")

            # --- PnL per trade ---
            pnl_col1, pnl_col2 = st.columns(2)

            with pnl_col1:
                st.markdown(f"##### {t('pnl_per_trade')}")
                fig_pnl = go.Figure()
                colors = [BINANCE_GREEN if p > 0 else BINANCE_RED for p in pnl_data["pnl_usd"]]
                fig_pnl.add_trace(go.Bar(
                    x=list(range(1, len(pnl_data) + 1)),
                    y=pnl_data["pnl_usd"],
                    marker_color=colors,
                    name=t("pnl_per_trade"),
                    hovertemplate="Trade #%{x}<br>PnL: $%{y:.2f}<extra></extra>",
                ))
                fig_pnl.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Trade #",
                    yaxis_title="PnL ($)",
                    hovermode="x unified",
                )
                apply_binance_theme(fig_pnl)
                st.plotly_chart(fig_pnl, use_container_width=True)

            with pnl_col2:
                st.markdown(f"##### {t('pnl_by_symbol')}")
                pnl_by_sym = pnl_data.groupby("symbol")["pnl_usd"].sum().sort_values()
                fig_sym = go.Figure()
                sym_colors = [BINANCE_GREEN if p > 0 else BINANCE_RED for p in pnl_by_sym.values]
                fig_sym.add_trace(go.Bar(
                    x=pnl_by_sym.index,
                    y=pnl_by_sym.values,
                    marker_color=sym_colors,
                    text=[f"${v:+.2f}" for v in pnl_by_sym.values],
                    textposition="outside",
                    textfont=dict(color="#EAECEF"),
                ))
                fig_sym.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="PnL ($)",
                )
                apply_binance_theme(fig_sym)
                st.plotly_chart(fig_sym, use_container_width=True)

            # --- Strategy & Coin stats (only if 3+ trades) ---
            if len(history) >= 3:
                st.divider()

                # Strategy stats
                if "strategy" in pnl_data.columns:
                    st.markdown(f"##### {t('strategy_stats')}")
                    strat_stats = pnl_data.groupby("strategy").agg(
                        trades=("pnl_usd", "count"),
                        total_pnl=("pnl_usd", "sum"),
                        avg_pnl=("pnl_usd", "mean"),
                        win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
                        max_win=("pnl_usd", "max"),
                        max_loss=("pnl_usd", "min"),
                    ).round(2)
                    strat_stats.columns = [
                        t("total_trades"), "Total PnL ($)", "Avg PnL ($)",
                        f"{t('win_rate')} (%)", "Max Win ($)", "Max Loss ($)",
                    ]
                    st.dataframe(strat_stats, use_container_width=True)

                    # Strategy PnL bar chart
                    fig_strat = go.Figure()
                    strat_pnl = pnl_data.groupby("strategy")["pnl_usd"].sum().sort_values(ascending=False)
                    strat_colors = [BINANCE_GREEN if v > 0 else BINANCE_RED for v in strat_pnl.values]
                    fig_strat.add_trace(go.Bar(
                        x=strat_pnl.index,
                        y=strat_pnl.values,
                        marker_color=strat_colors,
                        text=[f"${v:+.2f}" for v in strat_pnl.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_strat.update_layout(
                        title=t("pnl_by_strategy"),
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis_title="PnL ($)",
                    )
                    apply_binance_theme(fig_strat)
                    st.plotly_chart(fig_strat, use_container_width=True)

                st.divider()

                # Coin stats
                st.markdown(f"##### {t('coin_stats')}")
                coin_stats = pnl_data.groupby("symbol").agg(
                    trades=("pnl_usd", "count"),
                    total_pnl=("pnl_usd", "sum"),
                    avg_pnl=("pnl_usd", "mean"),
                    win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
                    max_win=("pnl_usd", "max"),
                    max_loss=("pnl_usd", "min"),
                ).round(2)
                coin_stats.columns = [
                    t("total_trades"), "Total PnL ($)", "Avg PnL ($)",
                    f"{t('win_rate')} (%)", "Max Win ($)", "Max Loss ($)",
                ]
                st.dataframe(coin_stats, use_container_width=True)

                # Coin PnL bar chart
                fig_coin = go.Figure()
                coin_pnl = pnl_data.groupby("symbol")["pnl_usd"].sum().sort_values(ascending=False)
                coin_colors = [BINANCE_GREEN if v > 0 else BINANCE_RED for v in coin_pnl.values]
                fig_coin.add_trace(go.Bar(
                    x=coin_pnl.index,
                    y=coin_pnl.values,
                    marker_color=coin_colors,
                    text=[f"${v:+.2f}" for v in coin_pnl.values],
                    textposition="outside",
                    textfont=dict(color="#EAECEF"),
                ))
                fig_coin.update_layout(
                    title=t("pnl_by_symbol"),
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="PnL ($)",
                )
                apply_binance_theme(fig_coin)
                st.plotly_chart(fig_coin, use_container_width=True)

                st.divider()

                # Exit reasons
                if "reason" in pnl_data.columns:
                    st.markdown(f"##### {t('exit_reasons')}")
                    exit_stats = pnl_data.groupby("reason").agg(
                        count=("pnl_usd", "count"),
                        total_pnl=("pnl_usd", "sum"),
                        avg_pnl=("pnl_usd", "mean"),
                        win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
                    ).round(2)
                    exit_stats.columns = [
                        "Count", "Total PnL ($)", "Avg PnL ($)", f"{t('win_rate')} (%)",
                    ]
                    st.dataframe(exit_stats, use_container_width=True)

                st.divider()

                # Signal log
                st.markdown(f"##### {t('signal_log')}")
                signal_df = load_signal_log()
                if not signal_df.empty:
                    # Filters
                    filt_col1, filt_col2, filt_col3 = st.columns(3)
                    with filt_col1:
                        actions = [t("all")] + sorted(signal_df["action"].unique().tolist()) if "action" in signal_df.columns else [t("all")]
                        sel_action = st.selectbox(t("filter_action"), actions, key="spot_sig_action")
                    with filt_col2:
                        symbols = [t("all")] + sorted(signal_df["symbol"].unique().tolist()) if "symbol" in signal_df.columns else [t("all")]
                        sel_symbol = st.selectbox(t("filter_symbol"), symbols, key="spot_sig_symbol")
                    with filt_col3:
                        strategies = [t("all")] + sorted(signal_df["strategy"].unique().tolist()) if "strategy" in signal_df.columns else [t("all")]
                        sel_strategy = st.selectbox(t("filter_strategy"), strategies, key="spot_sig_strategy")

                    filtered = signal_df.copy()
                    if sel_action != t("all") and "action" in filtered.columns:
                        filtered = filtered[filtered["action"] == sel_action]
                    if sel_symbol != t("all") and "symbol" in filtered.columns:
                        filtered = filtered[filtered["symbol"] == sel_symbol]
                    if sel_strategy != t("all") and "strategy" in filtered.columns:
                        filtered = filtered[filtered["strategy"] == sel_strategy]

                    st.dataframe(filtered.tail(50), use_container_width=True, hide_index=True)
                else:
                    st.info(t("no_signal_log"))
        else:
            st.info(t("no_trades"))

    # ─── FUTURES PERFORMANCE ──────────────────────────────────
    with perf_futures_tab:
        futures_state = load_futures_state()

        if futures_state is not None:
            f_history = futures_state.get("trade_history", [])

            if f_history and len(f_history) >= 1:
                f_pnl_data = pd.DataFrame(f_history)

                st.markdown(f"#### {t('pnl_analysis')} - Futures")

                # --- PnL per trade ---
                fp_col1, fp_col2 = st.columns(2)

                with fp_col1:
                    st.markdown(f"##### {t('pnl_per_trade')}")
                    fig_fpnl = go.Figure()
                    f_colors = [BINANCE_GREEN if p > 0 else BINANCE_RED for p in f_pnl_data["pnl_usd"]]
                    fig_fpnl.add_trace(go.Bar(
                        x=list(range(1, len(f_pnl_data) + 1)),
                        y=f_pnl_data["pnl_usd"],
                        marker_color=f_colors,
                        name=t("pnl_per_trade"),
                        hovertemplate="Trade #%{x}<br>PnL: $%{y:.2f}<extra></extra>",
                    ))
                    fig_fpnl.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Trade #",
                        yaxis_title="PnL ($)",
                    )
                    apply_binance_theme(fig_fpnl)
                    st.plotly_chart(fig_fpnl, use_container_width=True)

                with fp_col2:
                    # ROE per trade
                    st.markdown(f"##### {t('roe')} per Trade")
                    if "roe" in f_pnl_data.columns:
                        fig_roe = go.Figure()
                        roe_colors = [BINANCE_GREEN if r > 0 else BINANCE_RED for r in f_pnl_data["roe"]]
                        fig_roe.add_trace(go.Bar(
                            x=list(range(1, len(f_pnl_data) + 1)),
                            y=f_pnl_data["roe"],
                            marker_color=roe_colors,
                            name=t("roe"),
                            hovertemplate="Trade #%{x}<br>ROE: %{y:.2f}%<extra></extra>",
                        ))
                        fig_roe.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Trade #",
                            yaxis_title="ROE (%)",
                        )
                        apply_binance_theme(fig_roe)
                        st.plotly_chart(fig_roe, use_container_width=True)
                    else:
                        st.info(t("no_data"))

                st.divider()

                # --- PnL by symbol ---
                fp2_col1, fp2_col2 = st.columns(2)

                with fp2_col1:
                    st.markdown(f"##### {t('pnl_by_symbol')}")
                    f_pnl_sym = f_pnl_data.groupby("symbol")["pnl_usd"].sum().sort_values()
                    fig_fsym = go.Figure()
                    fsym_colors = [BINANCE_GREEN if p > 0 else BINANCE_RED for p in f_pnl_sym.values]
                    fig_fsym.add_trace(go.Bar(
                        x=f_pnl_sym.index,
                        y=f_pnl_sym.values,
                        marker_color=fsym_colors,
                        text=[f"${v:+.2f}" for v in f_pnl_sym.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_fsym.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title="PnL ($)",
                    )
                    apply_binance_theme(fig_fsym)
                    st.plotly_chart(fig_fsym, use_container_width=True)

                with fp2_col2:
                    # PnL Long vs Short
                    st.markdown(f"##### {t('pnl_long_short')}")
                    if "side" in f_pnl_data.columns:
                        side_pnl = f_pnl_data.groupby("side")["pnl_usd"].sum()
                        fig_side = go.Figure()
                        side_colors_map = {"long": BINANCE_GREEN, "short": BINANCE_RED, "LONG": BINANCE_GREEN, "SHORT": BINANCE_RED}
                        side_bar_colors = [side_colors_map.get(s, BINANCE_YELLOW) for s in side_pnl.index]
                        fig_side.add_trace(go.Bar(
                            x=side_pnl.index.str.upper(),
                            y=side_pnl.values,
                            marker_color=side_bar_colors,
                            text=[f"${v:+.2f}" for v in side_pnl.values],
                            textposition="outside",
                            textfont=dict(color="#EAECEF"),
                        ))
                        fig_side.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=10, b=0),
                            yaxis_title="PnL ($)",
                        )
                        apply_binance_theme(fig_side)
                        st.plotly_chart(fig_side, use_container_width=True)
                    else:
                        st.info(t("no_data"))

                st.divider()

                # --- Exit reasons ---
                if "reason" in f_pnl_data.columns:
                    st.markdown(f"##### {t('exit_reasons')}")
                    f_exit_stats = f_pnl_data.groupby("reason").agg(
                        count=("pnl_usd", "count"),
                        total_pnl=("pnl_usd", "sum"),
                        avg_pnl=("pnl_usd", "mean"),
                        win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
                    ).round(2)
                    f_exit_stats.columns = [
                        "Count", "Total PnL ($)", "Avg PnL ($)", f"{t('win_rate')} (%)",
                    ]
                    st.dataframe(f_exit_stats, use_container_width=True)

                st.divider()

                # --- Full trade history table ---
                st.markdown(f"##### {t('trade_history')}")
                display_f_cols = ["symbol", "side", "entry_price", "exit_price",
                                  "pnl_usd", "pnl_pct", "roe", "reason", "closed_at"]
                avail_f_cols = [c for c in display_f_cols if c in f_pnl_data.columns]
                if avail_f_cols:
                    display_f_df = f_pnl_data[avail_f_cols].copy()
                    if "side" in display_f_df.columns:
                        display_f_df["side"] = display_f_df["side"].str.upper()
                    st.dataframe(display_f_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(f_pnl_data, use_container_width=True, hide_index=True)
            else:
                st.info(t("no_trades"))
        else:
            st.info(t("no_futures_data"))


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.divider()

footer_left, footer_right = st.columns(2)
with footer_left:
    st.caption(
        f"{t('last_updated')}: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
with footer_right:
    st.caption(f"{t('auto_refresh')}: 5 min")

# Auto-refresh every 5 minutes (300 seconds)
st.markdown(
    """<meta http-equiv="refresh" content="300">""",
    unsafe_allow_html=True,
)
