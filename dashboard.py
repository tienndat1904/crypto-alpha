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

VN_TZ = timezone(timedelta(hours=7))

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

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
/* ===== HIDE DEFAULT STREAMLIT HEADER ===== */
header[data-testid="stHeader"] {
    display: none !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
.stApp > header {
    display: none !important;
}
/* Reduce top padding since header is hidden */
.stMainBlockContainer, div[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background-color: #0B0E11;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E2329 0%, #181A20 100%);
    border-right: 1px solid #2B3139;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #EAECEF;
}

/* ===== METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E2329 0%, #252930 100%);
    border: 1px solid #2B3139;
    border-radius: 12px;
    padding: 16px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"]:hover {
    border-color: #FCD535;
    box-shadow: 0 4px 16px rgba(252,213,53,0.1);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: #848E9C !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #EAECEF !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] > div {
    font-weight: 600 !important;
}

/* ===== TABS ===== */
div[data-testid="stTabs"] {
    background-color: transparent;
}
button[data-baseweb="tab"] {
    color: #848E9C !important;
    background-color: transparent !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #EAECEF !important;
    background-color: rgba(252,213,53,0.05) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FCD535 !important;
    border-bottom: 3px solid #FCD535 !important;
    font-weight: 600 !important;
}

/* ===== TABLES / DATAFRAMES ===== */
div[data-testid="stDataFrame"] {
    background-color: #1E2329;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2B3139;
}
div[data-testid="stDataFrame"] table thead tr {
    background-color: #2B3139;
}

/* ===== BUTTONS ===== */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #FCD535 0%, #F0B90B 100%) !important;
    color: #181A20 !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(252,213,53,0.2) !important;
}
button[kind="primary"]:hover, .stButton > button:hover {
    background: linear-gradient(135deg, #F0B90B 0%, #FCD535 100%) !important;
    box-shadow: 0 4px 16px rgba(252,213,53,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ===== STAR / TERTIARY BUTTONS ===== */
[data-testid="stBaseButton-tertiary"] > button,
button[kind="tertiary"] {
    background: #1E2329 !important;
    border: 1px solid #2B3139 !important;
    box-shadow: none !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    color: #EAECEF !important;
    min-height: 0 !important;
    height: 30px !important;
    line-height: 1 !important;
}
[data-testid="stBaseButton-tertiary"] > button:hover,
button[kind="tertiary"]:hover {
    background: #2B3139 !important;
    border-color: #FCD535 !important;
    color: #FCD535 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ===== GENERAL TEXT ===== */
.stMarkdown, .stText, p, span, label {
    color: #EAECEF;
}
h1, h2, h3, h4, h5, h6 {
    color: #EAECEF !important;
}
h1 {
    font-weight: 800 !important;
}

/* ===== DIVIDER ===== */
hr {
    border-color: rgba(43,49,57,0.5);
    margin: 24px 0;
}

/* ===== ALERTS ===== */
div[data-testid="stAlert"] {
    background: linear-gradient(135deg, #1E2329 0%, #252930 100%);
    border: 1px solid #2B3139;
    border-radius: 12px;
    color: #EAECEF;
}

/* ===== FORM ELEMENTS ===== */
div[data-baseweb="select"] {
    background-color: #1E2329;
    border: 1px solid #2B3139;
    border-radius: 8px;
    transition: all 0.2s ease;
}
div[data-baseweb="select"]:hover,
div[data-baseweb="select"]:focus-within {
    border-color: #FCD535;
    box-shadow: 0 0 0 1px rgba(252,213,53,0.2);
}
div[data-baseweb="input"] {
    background-color: #1E2329;
    border: 1px solid #2B3139;
    border-radius: 8px;
    transition: all 0.2s ease;
}
div[data-baseweb="input"]:hover,
div[data-baseweb="input"]:focus-within {
    border-color: #FCD535;
    box-shadow: 0 0 0 1px rgba(252,213,53,0.2);
}

/* ===== DROPDOWNS ===== */
div[data-baseweb="popover"] {
    background-color: #1E2329 !important;
    border: 1px solid #2B3139;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
ul[role="listbox"] {
    background-color: #1E2329 !important;
    border: 1px solid #2B3139;
}
ul[role="listbox"] li {
    color: #EAECEF !important;
    transition: background-color 0.15s ease;
}
ul[role="listbox"] li:hover {
    background-color: #2B3139 !important;
}
div[data-baseweb="select"] input {
    color: #EAECEF !important;
}
div[data-baseweb="select"] div[class*="ValueContainer"] {
    color: #EAECEF !important;
}
div[data-baseweb="select"] svg {
    fill: #848E9C;
}

/* ===== CUSTOM CLASSES ===== */
.hero-header {
    background: linear-gradient(135deg, #1E2329 0%, #0B0E11 50%, #1a1c24 100%);
    border: 1px solid #2B3139;
    border-radius: 16px;
    padding: 18px 24px 14px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FCD535 0%, #F0B90B 40%, #0ECB81 100%);
}
.hero-title {
    font-size: 22px;
    font-weight: 800;
    color: #EAECEF;
    margin: 0 0 2px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 14px;
    color: #848E9C;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(14,203,129,0.1);
    border: 1px solid rgba(14,203,129,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #0ECB81;
    font-weight: 600;
}
.live-dot {
    width: 8px; height: 8px;
    background: #0ECB81;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(14,203,129,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(14,203,129,0); }
}

.ticker-card {
    background: linear-gradient(135deg, #1E2329 0%, #252930 100%);
    border: 1px solid #2B3139;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.ticker-card:hover {
    border-color: #3B4049;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transform: translateY(-2px);
}
.ticker-card .coin-name {
    color: #EAECEF;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.ticker-card .coin-price {
    color: #EAECEF;
    font-size: 22px;
    font-weight: 700;
    margin: 6px 0 4px 0;
    letter-spacing: -0.5px;
}
.ticker-card .coin-change {
    font-size: 14px;
    font-weight: 600;
}
.ticker-card .coin-volume {
    color: #5E6673;
    font-size: 11px;
    margin-top: 6px;
}
.ticker-card .change-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
}
.change-badge.up {
    background: rgba(14,203,129,0.12);
    color: #0ECB81;
}
.change-badge.down {
    background: rgba(246,70,93,0.12);
    color: #F6465D;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #2B3139;
}
.section-header .section-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.section-header .section-title {
    font-size: 18px;
    font-weight: 700;
    color: #EAECEF;
    margin: 0;
}
.section-header .section-subtitle {
    font-size: 12px;
    color: #848E9C;
    margin: 0;
}

.summary-strip {
    display: flex;
    gap: 10px;
    padding: 8px 0 0;
    margin-bottom: 0;
    overflow-x: auto;
}
.summary-item {
    background: rgba(43,49,57,0.3);
    border: 1px solid #2B3139;
    border-radius: 8px;
    padding: 8px 16px;
    min-width: 120px;
    text-align: center;
    flex-shrink: 0;
}
.summary-item .label {
    font-size: 10px;
    color: #848E9C;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.summary-item .value {
    font-size: 16px;
    font-weight: 700;
    color: #EAECEF;
}

.pos-card {
    background: linear-gradient(135deg, #1E2329 0%, #252930 100%);
    border: 1px solid #2B3139;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
}
.pos-card:hover {
    border-color: #3B4049;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.pos-card .pos-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.pos-card .pos-symbol {
    font-size: 16px;
    font-weight: 700;
    color: #EAECEF;
}
.pos-card .pos-side {
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
}
.pos-card .pos-side.long {
    background: rgba(14,203,129,0.15);
    color: #0ECB81;
}
.pos-card .pos-side.short {
    background: rgba(246,70,93,0.15);
    color: #F6465D;
}
.pos-card .pos-detail {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 13px;
}
.pos-card .pos-detail .pos-label {
    color: #848E9C;
}
.pos-card .pos-detail .pos-value {
    color: #EAECEF;
    font-weight: 500;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0E11;
}
::-webkit-scrollbar-thumb {
    background: #2B3139;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3B4049;
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
        "current_price": "Giá hiện tại",
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
        "mtf_daily": "Xu hướng ngày",
        "mtf_hourly": "Xu hướng giờ",
        "mtf_confirmed": "MTF xác nhận",
        "mtf_filter": "Lọc MTF",
        "mtf_overview": "Tổng quan MTF",
        "mtf_confirmed_count": "Tín hiệu MTF xác nhận",
        "mtf_rejected_count": "Tín hiệu MTF từ chối",
        "mtf_confirmed_wr": "Win rate MTF xác nhận",
        "mtf_not_confirmed_wr": "Win rate không MTF",
        "mtf_bias": "Xu hướng MTF",
        "all_trades": "Tất cả giao dịch",
        "download_csv": "Tải CSV",
        "strategy_performance": "Hiệu suất theo chiến lược",
        "avg_pnl": "PnL trung bình",
        "avg_duration": "Thời gian giữ TB",
        "total": "Tổng",
        "drawdown_chart": "Biểu đồ Drawdown",
        "max_dd_point": "Drawdown lớn nhất",
        "current_equity": "Equity hiện tại",
        "unrealized": "Chưa thực hiện",
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
        "current_price": "Current Price",
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
        "mtf_daily": "Daily Trend",
        "mtf_hourly": "Hourly Trend",
        "mtf_confirmed": "MTF Confirmed",
        "mtf_filter": "MTF Filter",
        "mtf_overview": "MTF Overview",
        "mtf_confirmed_count": "MTF Confirmed Signals",
        "mtf_rejected_count": "MTF Rejected Signals",
        "mtf_confirmed_wr": "Win Rate (MTF Confirmed)",
        "mtf_not_confirmed_wr": "Win Rate (No MTF)",
        "mtf_bias": "MTF Bias",
        "all_trades": "All Trades",
        "download_csv": "Download CSV",
        "strategy_performance": "Strategy Performance",
        "avg_pnl": "Avg PnL",
        "avg_duration": "Avg Hold Time",
        "total": "Total",
        "drawdown_chart": "Drawdown Chart",
        "max_dd_point": "Max Drawdown",
        "current_equity": "Current Equity",
        "unrealized": "Unrealized",
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
BACKTEST_SPOT_FILE = Path("backtest_spot_results.csv")
BACKTEST_FUTURES_FILE = Path("backtest_futures_results.csv")
BACKTEST_FILE = Path("backtest_results.csv")  # legacy fallback
OPTIMIZED_FILE = Path("optimized_results.csv")
EQUITY_SPOT_JSON = Path("backtest_spot_equity.json")
EQUITY_FUTURES_JSON = Path("backtest_futures_equity.json")
EQUITY_JSON_FILE = Path("backtest_equity.json")  # legacy fallback
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
def load_backtest_results(mode="spot") -> pd.DataFrame:
    """Load backtest results CSV for spot or futures."""
    if mode == "spot":
        f = BACKTEST_SPOT_FILE if BACKTEST_SPOT_FILE.exists() else BACKTEST_FILE
    else:
        f = BACKTEST_FUTURES_FILE
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_optimized_results() -> pd.DataFrame:
    """Load optimized / walk-forward results CSV."""
    if OPTIMIZED_FILE.exists():
        return pd.read_csv(OPTIMIZED_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_equity_curves(mode="spot") -> dict:
    """Load backtest equity curves from JSON for spot or futures."""
    if mode == "spot":
        f = EQUITY_SPOT_JSON if EQUITY_SPOT_JSON.exists() else EQUITY_JSON_FILE
    else:
        f = EQUITY_FUTURES_JSON
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
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


WATCHED_COINS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT",
    "DOT/USDT", "LINK/USDT", "POL/USDT", "UNI/USDT",
    "ATOM/USDT", "LTC/USDT", "FIL/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "NEAR/USDT", "INJ/USDT",
]


@st.cache_data(ttl=900)
def fetch_tickers():
    """Fetch market tickers from Binance via ccxt."""
    if ccxt is None:
        return {}
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        tickers = exchange.fetch_tickers(WATCHED_COINS)
        return tickers
    except Exception as e:
        logger.warning(f"fetch_tickers error: {e}")
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

# Auto-refresh every 15 minutes (900000ms)
if st_autorefresh is not None:
    st_autorefresh(interval=900_000, limit=None, key="auto_refresh_15m")

# Language selector (small, top-right)
_lang_col1, _lang_col2 = st.columns([10, 1])
with _lang_col2:
    lang_options = {"VI": "vi", "EN": "en"}
    selected_lang_label = st.selectbox(
        "Lang",
        options=list(lang_options.keys()),
        index=0 if st.session_state.lang == "vi" else 1,
        label_visibility="collapsed",
    )
    new_lang = lang_options[selected_lang_label]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

# ── Hero Header ──
_now_str = datetime.now(VN_TZ).strftime("%H:%M (VN)")

# Load Spot state
_spot_state = load_state()
_spot_equity = _spot_state["capital"]
for _p in _spot_state.get("open_positions", {}).values():
    _spot_equity += _p.get("size_usdt", 0)
_spot_pnl = _spot_state.get("total_pnl", 0)
_spot_trades = _spot_state.get("total_trades", 0)
_spot_wins = _spot_state.get("total_wins", 0)
_spot_positions = len(_spot_state.get("open_positions", {}))

# Load Futures state
_fut_state = load_futures_state()
if _fut_state:
    _fut_equity = _fut_state.get("capital", 0)
    for _fp in _fut_state.get("open_positions", {}).values():
        _fut_equity += _fp.get("margin", _fp.get("size_usdt", 0) / _fp.get("leverage", 3))
    _fut_pnl = _fut_state.get("total_pnl", 0)
    _fut_trades = _fut_state.get("total_trades", 0)
    _fut_wins = _fut_state.get("total_wins", 0)
    _fut_positions = len(_fut_state.get("open_positions", {}))
    _fut_leverage = 3
    for _fp in _fut_state.get("open_positions", {}).values():
        _fut_leverage = _fp.get("leverage", 3)
        break
else:
    _fut_equity = 0
    _fut_pnl = 0
    _fut_trades = 0
    _fut_wins = 0
    _fut_positions = 0
    _fut_leverage = 3

# Totals
_total_equity = _spot_equity + _fut_equity
_total_pnl = _spot_pnl + _fut_pnl
_total_trades = _spot_trades + _fut_trades
_total_wins = _spot_wins + _fut_wins
_total_positions = _spot_positions + _fut_positions

# View selector
_hero_col1, _hero_col2 = st.columns([6, 1])
with _hero_col2:
    _view_mode = st.radio(
        "View",
        ["Total", "Spot", "Futures"],
        horizontal=True,
        label_visibility="collapsed",
        key="hero_view_mode",
    )

# Pick data based on view
if _view_mode == "Spot":
    _hero_equity = _spot_equity
    _hero_pnl = _spot_pnl
    _hero_trades = _spot_trades
    _hero_wins = _spot_wins
    _hero_positions = _spot_positions
    _hero_max_pos = 3
    _hero_badge = "SPOT"
elif _view_mode == "Futures":
    _hero_equity = _fut_equity
    _hero_pnl = _fut_pnl
    _hero_trades = _fut_trades
    _hero_wins = _fut_wins
    _hero_positions = _fut_positions
    _hero_max_pos = 3
    _hero_badge = f"FUTURES {_fut_leverage}x"
else:
    _hero_equity = _total_equity
    _hero_pnl = _total_pnl
    _hero_trades = _total_trades
    _hero_wins = _total_wins
    _hero_positions = _total_positions
    _hero_max_pos = 6
    _hero_badge = "TOTAL"

_hero_wr = (_hero_wins / _hero_trades * 100) if _hero_trades > 0 else 0
_hero_pnl_color = BINANCE_GREEN if _hero_pnl >= 0 else BINANCE_RED
_hero_pnl_sign = "+" if _hero_pnl >= 0 else ""

# Spot vs Futures breakdown line
if _view_mode == "Total" and _fut_state:
    _spot_color = BINANCE_GREEN if _spot_pnl >= 0 else BINANCE_RED
    _fut_color = BINANCE_GREEN if _fut_pnl >= 0 else BINANCE_RED
    _breakdown_html = (
        f'<div style="display:flex;gap:20px;margin-top:6px;font-size:11px;color:#848E9C;">'
        f'<span>Spot: <span style="color:#EAECEF;">${_spot_equity:,.2f}</span> '
        f'(<span style="color:{_spot_color};">{"+" if _spot_pnl >= 0 else ""}${_spot_pnl:,.2f}</span>)</span>'
        f'<span>Futures: <span style="color:#EAECEF;">${_fut_equity:,.2f}</span> '
        f'(<span style="color:{_fut_color};">{"+" if _fut_pnl >= 0 else ""}${_fut_pnl:,.2f}</span>)</span>'
        f'</div>'
    )
else:
    _breakdown_html = ""

with _hero_col1:
    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <p class="hero-title">Crypto Alpha</p>
                <p class="hero-subtitle">{t("subtitle")}</p>
            </div>
            <div style="text-align:right;">
                <div class="hero-badge"><span class="live-dot"></span> {_hero_badge}</div>
                <div style="color:#5E6673;font-size:11px;margin-top:6px;">{_now_str}</div>
            </div>
        </div>
        <div class="summary-strip">
            <div class="summary-item">
                <div class="label">{t("equity")}</div>
                <div class="value">${_hero_equity:,.2f}</div>
            </div>
            <div class="summary-item">
                <div class="label">PnL</div>
                <div class="value" style="color:{_hero_pnl_color};">{_hero_pnl_sign}${_hero_pnl:,.2f}</div>
            </div>
            <div class="summary-item">
                <div class="label">{t("total_trades")}</div>
                <div class="value">{_hero_trades}</div>
            </div>
            <div class="summary-item">
                <div class="label">{t("win_rate")}</div>
                <div class="value">{_hero_wr:.0f}%</div>
            </div>
            <div class="summary-item">
                <div class="label">{t("open_positions")}</div>
                <div class="value">{_hero_positions}/{_hero_max_pos}</div>
            </div>
        </div>
        {_breakdown_html}
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon" style="background:rgba(252,213,53,0.1);">
            <span style="color:#FCD535;">$</span>
        </div>
        <div>
            <p class="section-title">{t("market_watch")}</p>
            <p class="section-subtitle">{t("overview_24h")}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Favorites system (persist to file) ---
    FAVORITES_FILE = Path("trading/favorites.json")

    def _load_favorites():
        if FAVORITES_FILE.exists():
            try:
                return json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

    def _save_favorites(favs):
        FAVORITES_FILE.parent.mkdir(parents=True, exist_ok=True)
        FAVORITES_FILE.write_text(json.dumps(favs, indent=2), encoding="utf-8")

    if "favorites" not in st.session_state:
        st.session_state["favorites"] = _load_favorites()

    def _toggle_fav(coin):
        favs = st.session_state["favorites"]
        if coin in favs:
            favs.remove(coin)
        else:
            favs.append(coin)
        st.session_state["favorites"] = favs
        _save_favorites(favs)

    # --- View toggle: Favorites / All ---
    fav_view_col, fav_count_col = st.columns([3, 1])
    with fav_view_col:
        show_mode = st.radio(
            "View",
            options=["favorites", "all"],
            format_func=lambda x: f"Favorites ({len(st.session_state['favorites'])})" if x == "favorites" else f"All ({len(WATCHED_COINS)})",
            horizontal=True,
            key="market_view_mode",
            label_visibility="collapsed",
        )

    # --- Star selector row ---
    display_coins = st.session_state["favorites"] if show_mode == "favorites" else WATCHED_COINS
    star_container = st.container()
    with star_container:
        star_cols = st.columns(10)
        for idx, coin in enumerate(WATCHED_COINS):
            coin_short = coin.split("/")[0]
            is_fav = coin in st.session_state["favorites"]
            star = "★" if is_fav else "☆"
            with star_cols[idx % 10]:
                if st.button(
                    f"{star} {coin_short}",
                    key=f"star_{coin}",
                    use_container_width=True,
                    type="tertiary",
                ):
                    _toggle_fav(coin)
                    st.rerun()

    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    # --- Ticker cards ---
    tickers_data = fetch_tickers()

    if tickers_data and display_coins:
        ticker_cols = st.columns(4)
        for idx, coin in enumerate(display_coins):
            tk = tickers_data.get(coin, None)
            if tk is None:
                continue
            col = ticker_cols[idx % 4]
            last_price = tk.get("last", 0)
            change_pct = tk.get("percentage", 0) or 0
            vol_24h = tk.get("quoteVolume", 0) or 0
            sign = "+" if change_pct >= 0 else ""
            badge_cls = "up" if change_pct >= 0 else "down"
            arrow = "^" if change_pct >= 0 else "v"
            coin_short = coin.split("/")[0]
            is_fav = coin in st.session_state["favorites"]
            star_icon = "★" if is_fav else ""
            star_html = f'<span style="color:#FCD535;font-size:14px;margin-right:4px;">{star_icon}</span>' if is_fav else ""

            # Format volume nicely
            if vol_24h >= 1_000_000_000:
                vol_str = f"${vol_24h/1_000_000_000:.1f}B"
            elif vol_24h >= 1_000_000:
                vol_str = f"${vol_24h/1_000_000:.1f}M"
            else:
                vol_str = f"${vol_24h:,.0f}"

            # Format price based on magnitude
            if last_price >= 100:
                price_str = f"${last_price:,.2f}"
            elif last_price >= 1:
                price_str = f"${last_price:,.3f}"
            else:
                price_str = f"${last_price:,.4f}"

            with col:
                st.markdown(
                    f"""
                    <div class="ticker-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span class="coin-name">{star_html}{coin_short}</span>
                            <span class="change-badge {badge_cls}">{arrow} {sign}{change_pct:.2f}%</span>
                        </div>
                        <div class="coin-price">{price_str}</div>
                        <div class="coin-volume">Vol {vol_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # --- Candlestick chart ---
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon" style="background:rgba(14,203,129,0.1);">
                <span style="color:#0ECB81;">W</span>
            </div>
            <div><p class="section-title">{t("candlestick_chart")}</p></div>
        </div>
        """, unsafe_allow_html=True)
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
                options=["15m", "1h", "4h", "1d"],
                index=1,
                key="candle_period",
            )

        ohlcv_df = fetch_ohlcv(selected_coin, selected_period, limit=100)
        if not ohlcv_df.empty:
            fig_candle = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
            )

            # Candlestick
            fig_candle.add_trace(go.Candlestick(
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
            ), row=1, col=1)

            # Volume bars
            vol_colors = [
                BINANCE_GREEN if c >= o else BINANCE_RED
                for c, o in zip(ohlcv_df["close"], ohlcv_df["open"])
            ]
            fig_candle.add_trace(go.Bar(
                x=ohlcv_df["timestamp"],
                y=ohlcv_df["volume"],
                marker_color=vol_colors,
                marker_opacity=0.4,
                name="Volume",
                showlegend=False,
            ), row=2, col=1)

            fig_candle.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                showlegend=False,
            )
            fig_candle.update_yaxes(title_text=t("price"), row=1, col=1, gridcolor="#2B3139")
            fig_candle.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="#2B3139")
            fig_candle.update_xaxes(gridcolor="#2B3139", row=1, col=1)
            fig_candle.update_xaxes(gridcolor="#2B3139", row=2, col=1)
            apply_binance_theme(fig_candle)
            st.plotly_chart(fig_candle, use_container_width=True)
        else:
            st.info(t("no_data"))

        st.divider()

        # --- Sparklines ---
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon" style="background:rgba(56,97,251,0.1);">
                <span style="color:#3861FB;">~</span>
            </div>
            <div><p class="section-title">{t("sparklines")}</p><p class="section-subtitle">48h trend</p></div>
        </div>
        """, unsafe_allow_html=True)
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
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon" style="background:rgba(14,203,129,0.1);">
                <span style="color:#0ECB81;">S</span>
            </div>
            <div><p class="section-title">{t("spot_trading")}</p></div>
        </div>
        """, unsafe_allow_html=True)
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
            st.markdown(f"""
            <div class="section-header" style="margin-top:0;">
                <div><p class="section-title" style="font-size:16px;">{t("open_positions")}</p></div>
            </div>
            """, unsafe_allow_html=True)
            # Fetch current prices for open positions
            _spot_current_prices = {}
            if open_pos and ccxt is not None:
                try:
                    _spot_ex = ccxt.binance({"enableRateLimit": True})
                    for _sym in open_pos:
                        try:
                            _tk = _spot_ex.fetch_ticker(_sym)
                            _spot_current_prices[_sym] = _tk["last"]
                        except Exception:
                            pass
                except Exception:
                    pass

            if open_pos:
                for sym, pos in open_pos.items():
                    side = pos["side"]
                    side_cls = "long" if side == "long" else "short"
                    side_label = side.upper()
                    entry_p = pos['entry_price']
                    stop_p = pos['stop_price']
                    tp1_str = f"${pos['tp1_price']:,.4f}" if "tp1_price" in pos else "-"
                    size_str = f"${pos.get('size_usdt', 0):.2f}"
                    opened = pos.get("opened_at", "")[:16]

                    # Current price & PnL (spot = LONG only)
                    cur_price = _spot_current_prices.get(sym)
                    if cur_price and entry_p > 0:
                        pnl_pct = (cur_price - entry_p) / entry_p * 100
                        pnl_usd = pos.get("size_usdt", 0) * pnl_pct / 100
                        pnl_color = BINANCE_GREEN if pnl_pct >= 0 else BINANCE_RED
                        price_html = f'<span class="pos-value">${{cur_price:,.4f}}</span>'
                        pnl_html = f'<span class="pos-value" style="color:{pnl_color};font-weight:700;">{pnl_pct:+.2f}% (${pnl_usd:+.2f})</span>'
                    else:
                        price_html = '<span class="pos-value" style="color:#848E9C;">--</span>'
                        pnl_html = '<span class="pos-value" style="color:#848E9C;">--</span>'

                    st.markdown(f"""
                    <div class="pos-card">
                        <div class="pos-header">
                            <span class="pos-symbol">{sym}</span>
                            <span class="pos-side {side_cls}">{side_label}</span>
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">{t("entry")}</span>
                            <span class="pos-value">${entry_p:,.4f}</span>
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">{t("current_price")}</span>
                            {price_html}
                        </div>
                        <div class="pos-detail" style="background:rgba({'14,203,129' if cur_price and pnl_pct >= 0 else '246,70,93'},0.05);border-radius:4px;padding:4px 8px;margin:2px 0;">
                            <span class="pos-label">PnL</span>
                            {pnl_html}
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">{t("size")}</span>
                            <span class="pos-value">{size_str}</span>
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">{t("stop")}</span>
                            <span class="pos-value" style="color:{BINANCE_RED};">${stop_p:,.4f}</span>
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">TP1</span>
                            <span class="pos-value" style="color:{BINANCE_GREEN};">{tp1_str}</span>
                        </div>
                        <div class="pos-detail">
                            <span class="pos-label">{t("opened_at")}</span>
                            <span class="pos-value" style="color:#848E9C;font-size:12px;">{opened}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#1E2329;border:1px solid #2B3139;border-radius:12px;
                            padding:40px;text-align:center;color:#848E9C;font-size:14px;">
                    {t("no_open_positions")}
                </div>
                """, unsafe_allow_html=True)

        # --- Charts for Open Positions ---
        if open_pos:
            st.markdown(f"""
            <div class="section-header">
                <div class="section-icon" style="background:rgba(252,213,53,0.1);">
                    <span style="color:#FCD535;">C</span>
                </div>
                <div><p class="section-title">{t("candlestick_chart")} - {t("open_positions")}</p></div>
            </div>
            """, unsafe_allow_html=True)
            chart_cols = st.columns(min(len(open_pos), 2))
            for idx, (sym, pos) in enumerate(open_pos.items()):
                with chart_cols[idx % min(len(open_pos), 2)]:
                    pair = sym if "/" in sym else f"{sym}/USDT"
                    ohlcv_pos = fetch_ohlcv(pair, "4h", limit=60)
                    if not ohlcv_pos.empty:
                        fig_pos = go.Figure(data=[go.Candlestick(
                            x=ohlcv_pos["timestamp"],
                            open=ohlcv_pos["open"], high=ohlcv_pos["high"],
                            low=ohlcv_pos["low"], close=ohlcv_pos["close"],
                            increasing_line_color=BINANCE_GREEN, decreasing_line_color=BINANCE_RED,
                            increasing_fillcolor=BINANCE_GREEN, decreasing_fillcolor=BINANCE_RED,
                        )])
                        entry_p = pos["entry_price"]
                        stop_p = pos["stop_price"]
                        side_label = pos["side"].upper()
                        fig_pos.add_hline(y=entry_p, line_dash="dash", line_color=BINANCE_YELLOW,
                                          annotation_text=f"Entry ${entry_p:,.2f}", annotation_font_color=BINANCE_YELLOW)
                        fig_pos.add_hline(y=stop_p, line_dash="dash", line_color=BINANCE_RED,
                                          annotation_text=f"Stop ${stop_p:,.2f}", annotation_font_color=BINANCE_RED)
                        if "tp1_price" in pos:
                            fig_pos.add_hline(y=pos["tp1_price"], line_dash="dash", line_color=BINANCE_GREEN,
                                              annotation_text=f"TP1 ${pos['tp1_price']:,.2f}", annotation_font_color=BINANCE_GREEN)
                        current_price = ohlcv_pos["close"].iloc[-1]
                        pnl_pct = ((current_price - entry_p) / entry_p * 100) if pos["side"] == "long" else ((entry_p - current_price) / entry_p * 100)
                        pnl_color = BINANCE_GREEN if pnl_pct >= 0 else BINANCE_RED
                        fig_pos.update_layout(
                            title=dict(text=f"{sym} [{side_label}] | Now: ${current_price:,.2f} ({pnl_pct:+.2f}%)",
                                       font=dict(color=pnl_color, size=14)),
                            height=350, margin=dict(l=0, r=0, t=35, b=0),
                            xaxis_rangeslider_visible=False,
                        )
                        apply_binance_theme(fig_pos)
                        st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                        st.warning(f"Cannot load chart for {sym}")
            st.divider()

        # --- All Trades (scrollable with download) ---
        with right_col:
            st.markdown(f"""
            <div class="section-header" style="margin-top:0;">
                <div><p class="section-title" style="font-size:16px;">{t("all_trades")}</p></div>
            </div>
            """, unsafe_allow_html=True)
            if history:
                hist_rows = []
                total_pnl_usd_sum = 0.0
                total_pnl_pct_sum = 0.0
                total_size_sum = 0.0
                for trade in history:
                    duration_str = "-"
                    if trade.get("duration_hours") is not None:
                        hrs = int(trade["duration_hours"])
                        mins = int((trade["duration_hours"] - hrs) * 60)
                        duration_str = f"{hrs}h {mins}m"
                    size_str = f"${trade.get('size_usdt', 0):.2f}" if trade.get("size_usdt") else "-"
                    total_pnl_usd_sum += trade["pnl_usd"]
                    total_pnl_pct_sum += trade["pnl_pct"]
                    total_size_sum += trade.get("size_usdt", 0)
                    hist_rows.append({
                        t("symbol"): trade["symbol"],
                        t("side"): trade["side"].upper(),
                        t("entry"): f"${trade['entry_price']:,.4f}",
                        t("exit"): f"${trade['exit_price']:,.4f}",
                        "PnL %": f"{trade['pnl_pct']:+.2f}%",
                        "PnL $": f"${trade['pnl_usd']:+.2f}",
                        t("reason"): trade["reason"],
                        "Strategy": trade.get("strategy", "-"),
                        t("size"): size_str,
                        t("avg_duration"): duration_str,
                    })
                # Summary row
                hist_rows.append({
                    t("symbol"): f"📊 {t('total')}",
                    t("side"): f"{len(history)} trades",
                    t("entry"): "",
                    t("exit"): "",
                    "PnL %": f"{total_pnl_pct_sum:+.2f}%",
                    "PnL $": f"${total_pnl_usd_sum:+.2f}",
                    t("reason"): "",
                    "Strategy": "",
                    t("size"): f"${total_size_sum:.2f}",
                    t("avg_duration"): "",
                })

                # Build HTML table with colored PnL
                html_header = "".join(f"<th style='padding:6px 10px;text-align:left;color:#848E9C;border-bottom:1px solid #2B3139;'>{col}</th>" for col in hist_rows[0].keys())
                html_rows = ""
                for i, row in enumerate(hist_rows):
                    bg = "#1E2329" if i % 2 == 0 else "#181A20"
                    if i == len(hist_rows) - 1:
                        bg = "#2B3139"  # summary row
                    cells = ""
                    for key, val in row.items():
                        color = "#EAECEF"
                        if key in ("PnL %", "PnL $") and val:
                            color = BINANCE_GREEN if "+" in str(val) else BINANCE_RED
                        cells += f"<td style='padding:5px 10px;color:{color};white-space:nowrap;'>{val}</td>"
                    html_rows += f"<tr style='background-color:{bg};'>{cells}</tr>"

                trade_table_html = f"""
                <div style="max-height:400px;overflow-y:auto;border:1px solid #2B3139;border-radius:8px;">
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <thead><tr style="background-color:#2B3139;position:sticky;top:0;">{html_header}</tr></thead>
                <tbody>{html_rows}</tbody>
                </table></div>
                """
                st.markdown(trade_table_html, unsafe_allow_html=True)

                # Download CSV button
                csv_df = pd.DataFrame(hist_rows[:-1])  # exclude summary row
                csv_data = csv_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇ {t('download_csv')}",
                    data=csv_data,
                    file_name="trade_history.csv",
                    mime="text/csv",
                )
            else:
                st.info(t("no_trades"))

        st.divider()

        # --- Equity Curve with Drawdown Subplot ---
        st.markdown(f"""
        <div class="section-header">
            <div class="section-icon" style="background:rgba(252,213,53,0.1);">
                <span style="color:#FCD535;">E</span>
            </div>
            <div><p class="section-title">{t("equity_curve")}</p></div>
        </div>
        """, unsafe_allow_html=True)
        if history:
            equity_points = [{"date": state.get("created_at", "2024-01-01"), "equity": 500}]
            running_equity = 500
            for trade in history:
                running_equity += trade["pnl_usd"]
                equity_points.append({
                    "date": trade.get("closed_at", ""),
                    "equity": running_equity,
                })

            eq_df = pd.DataFrame(equity_points)
            eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
            eq_df = eq_df.dropna()

            # Calculate drawdown from peak
            eq_df["peak"] = eq_df["equity"].cummax()
            eq_df["drawdown_pct"] = ((eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]) * 100

            # Max drawdown point
            max_dd_idx = eq_df["drawdown_pct"].idxmin()
            max_dd_date = eq_df.loc[max_dd_idx, "date"]
            max_dd_value = eq_df.loc[max_dd_idx, "drawdown_pct"]

            # Unrealized PnL from open positions
            unrealized_pnl = total_equity - running_equity

            # Create subplots: equity curve on top, drawdown below
            fig_eq = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=[t("equity_curve"), t("drawdown_chart")],
            )

            # Equity curve trace
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["equity"],
                mode="lines+markers",
                line=dict(color=BINANCE_YELLOW, width=2),
                marker=dict(size=6, color=BINANCE_YELLOW),
                name=t("equity"),
            ), row=1, col=1)

            # Unrealized PnL dashed extension line
            if abs(unrealized_pnl) > 0.001:
                last_trade_date = eq_df["date"].iloc[-1]
                current_date = pd.Timestamp(datetime.now(timezone.utc))
                unreal_color = BINANCE_GREEN if unrealized_pnl >= 0 else BINANCE_RED
                fig_eq.add_trace(go.Scatter(
                    x=[last_trade_date, current_date],
                    y=[running_equity, total_equity],
                    mode="lines+markers",
                    line=dict(color=unreal_color, width=2, dash="dash"),
                    marker=dict(size=8, color=unreal_color, symbol="diamond"),
                    name=f"{t('unrealized')}: ${unrealized_pnl:+.2f}",
                    hovertemplate=f"{t('current_equity')}: $%{{y:.2f}}<extra>{t('unrealized')}</extra>",
                ), row=1, col=1)

            # Initial capital line
            fig_eq.add_hline(
                y=500, line_dash="dash", line_color="#848E9C",
                annotation_text=f"{t('initial_capital')} $500",
                annotation_font_color="#848E9C",
                row=1, col=1,
            )

            # Drawdown area chart (negative values, red fill)
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["drawdown_pct"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(246, 70, 93, 0.3)",
                line=dict(color=BINANCE_RED, width=1),
                name=t("drawdown"),
                hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
            ), row=2, col=1)

            # Annotate max drawdown point
            fig_eq.add_annotation(
                x=max_dd_date,
                y=max_dd_value,
                text=f"{t('max_dd_point')}: {max_dd_value:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=BINANCE_RED,
                font=dict(color=BINANCE_RED, size=11),
                bgcolor="#1E2329",
                bordercolor=BINANCE_RED,
                borderwidth=1,
                row=2, col=1,
            )

            eq_min = eq_df["equity"].min()
            eq_max_val = max(eq_df["equity"].max(), total_equity)
            eq_pad = max((eq_max_val - eq_min) * 0.2, 2)
            fig_eq.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_eq.update_yaxes(title_text=t("equity") + " ($)", range=[eq_min - eq_pad, eq_max_val + eq_pad], row=1, col=1)
            fig_eq.update_yaxes(title_text="DD %", row=2, col=1)
            fig_eq.update_xaxes(gridcolor="#2B3139", zerolinecolor="#2B3139", row=1, col=1)
            fig_eq.update_xaxes(gridcolor="#2B3139", zerolinecolor="#2B3139", row=2, col=1)
            fig_eq.update_yaxes(gridcolor="#2B3139", zerolinecolor="#2B3139", row=1, col=1)
            fig_eq.update_yaxes(gridcolor="#2B3139", zerolinecolor="#2B3139", row=2, col=1)
            apply_binance_theme(fig_eq)
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info(t("equity_after_trade"))

        # --- Per-Strategy Performance ---
        if history:
            st.divider()
            st.markdown(f"""
            <div class="section-header">
                <div class="section-icon" style="background:rgba(224,64,251,0.1);">
                    <span style="color:#E040FB;">S</span>
                </div>
                <div><p class="section-title">{t("strategy_performance")}</p></div>
            </div>
            """, unsafe_allow_html=True)

            # Group trades by strategy
            strategy_map = {}
            for trade in history:
                strat = trade.get("strategy", "Unknown")
                if strat not in strategy_map:
                    strategy_map[strat] = []
                strategy_map[strat].append(trade)

            # Render strategy cards
            strat_names = list(strategy_map.keys())
            cols_per_row = min(len(strat_names), 4)
            strat_cols = st.columns(cols_per_row) if cols_per_row > 0 else []

            strategy_summary = []
            for idx, (strat_name, trades) in enumerate(strategy_map.items()):
                col = strat_cols[idx % cols_per_row]
                n_trades = len(trades)
                n_wins = sum(1 for t_ in trades if t_["pnl_usd"] > 0)
                wr = (n_wins / n_trades * 100) if n_trades > 0 else 0
                avg_pnl = sum(t_["pnl_usd"] for t_ in trades) / n_trades if n_trades > 0 else 0
                total_pnl_strat = sum(t_["pnl_usd"] for t_ in trades)
                durations = [t_.get("duration_hours", 0) for t_ in trades if t_.get("duration_hours") is not None]
                avg_dur = sum(durations) / len(durations) if durations else 0
                avg_dur_h = int(avg_dur)
                avg_dur_m = int((avg_dur - avg_dur_h) * 60)
                dur_str = f"{avg_dur_h}h {avg_dur_m}m" if durations else "-"

                strategy_summary.append({
                    "name": strat_name,
                    "trades": n_trades,
                    "win_rate": wr,
                    "avg_pnl": avg_pnl,
                    "total_pnl": total_pnl_strat,
                })

                header_color = BINANCE_GREEN if total_pnl_strat >= 0 else BINANCE_RED
                pnl_sign = "+" if total_pnl_strat >= 0 else ""
                avg_sign = "+" if avg_pnl >= 0 else ""

                card_html = f"""
                <div style="background-color:#1E2329;border:1px solid #2B3139;border-radius:8px;
                            padding:0;margin-bottom:8px;overflow:hidden;">
                    <div style="background-color:{header_color};padding:8px 12px;">
                        <span style="color:#181A20;font-weight:700;font-size:14px;">{strat_name}</span>
                    </div>
                    <div style="padding:10px 12px;font-size:13px;color:#EAECEF;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="color:#848E9C;">{t('total_trades')}</span>
                            <span>{n_trades}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="color:#848E9C;">{t('win_rate')}</span>
                            <span style="color:{BINANCE_GREEN if wr >= 50 else BINANCE_RED};">{wr:.1f}%</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="color:#848E9C;">{t('avg_pnl')}</span>
                            <span style="color:{BINANCE_GREEN if avg_pnl >= 0 else BINANCE_RED};">{avg_sign}${avg_pnl:.2f}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="color:#848E9C;">{t('total_return')}</span>
                            <span style="color:{BINANCE_GREEN if total_pnl_strat >= 0 else BINANCE_RED};">{pnl_sign}${total_pnl_strat:.2f}</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;">
                            <span style="color:#848E9C;">{t('avg_duration')}</span>
                            <span>{dur_str}</span>
                        </div>
                    </div>
                </div>
                """
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

            # Win rate comparison bar chart
            if strategy_summary:
                st.markdown(f"###### {t('win_rate')} — {t('strategy_compare')}")
                fig_wr = go.Figure()
                s_names = [s["name"] for s in strategy_summary]
                s_wr = [s["win_rate"] for s in strategy_summary]
                s_colors = [BINANCE_GREEN if w >= 50 else BINANCE_RED for w in s_wr]
                fig_wr.add_trace(go.Bar(
                    x=s_names,
                    y=s_wr,
                    marker_color=s_colors,
                    text=[f"{w:.1f}%" for w in s_wr],
                    textposition="outside",
                    textfont=dict(color="#EAECEF"),
                ))
                fig_wr.add_hline(y=50, line_dash="dash", line_color="#848E9C")
                fig_wr.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("win_rate") + " %",
                    yaxis_range=[0, max(max(s_wr) + 15, 60)],
                    showlegend=False,
                )
                apply_binance_theme(fig_wr)
                st.plotly_chart(fig_wr, use_container_width=True)

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
            f_consec_losses = futures_state.get("consecutive_losses", 0)

            # Calculate margin used and equity from open positions
            f_margin_used = 0
            f_total_equity = f_capital
            f_leverage = 3  # default
            for pos in f_open_pos.values():
                margin = pos.get("margin", pos.get("size_usdt", 0) / pos.get("leverage", 3))
                f_margin_used += margin
                f_total_equity += margin
                f_leverage = pos.get("leverage", 3)
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

            # --- Fetch current prices for futures positions ---
            _fut_current_prices = {}
            if f_open_pos and ccxt is not None:
                try:
                    _fut_ex = ccxt.binance({"enableRateLimit": True})
                    for _sym in f_open_pos:
                        try:
                            _tk = _fut_ex.fetch_ticker(_sym)
                            _fut_current_prices[_sym] = _tk["last"]
                        except Exception:
                            pass
                except Exception:
                    pass

            # --- Open Positions ---
            st.markdown(f"##### {t('open_positions')}")
            if f_open_pos:
                f_pos_rows = []
                for sym, pos in f_open_pos.items():
                    entry_p = pos.get("entry_price", 0)
                    size_u = pos.get("size_usdt", 0)  # notional
                    lev = pos.get("leverage", 3)
                    margin_val = pos.get("margin", size_u / lev)
                    liq_p = pos.get("liq_price", 0)
                    tp1_val = f"${pos['tp1_price']:,.4f}" if "tp1_price" in pos else "-"
                    tp2_val = f"${pos['tp2_price']:,.4f}" if "tp2_price" in pos else "-"

                    # Current price & PnL
                    cur_p = _fut_current_prices.get(sym)
                    if cur_p and entry_p > 0:
                        if pos["side"] == "long":
                            f_pnl_pct = (cur_p - entry_p) / entry_p * 100
                        else:
                            f_pnl_pct = (entry_p - cur_p) / entry_p * 100
                        f_roe = f_pnl_pct * lev
                        f_pnl_usd = margin_val * f_roe / 100
                        cur_p_str = f"${cur_p:,.4f}"
                        pnl_sign = "+" if f_pnl_pct >= 0 else ""
                        pnl_str = f"{pnl_sign}{f_pnl_pct:.2f}% (${f_pnl_usd:+.2f})"
                        roe_str = f"{pnl_sign}{f_roe:.2f}%"
                    else:
                        cur_p_str = "--"
                        pnl_str = "--"
                        roe_str = "--"

                    row = {
                        t("symbol"): sym,
                        t("side"): pos["side"].upper(),
                        t("entry"): f"${entry_p:,.4f}",
                        t("current_price"): cur_p_str,
                        "PnL": pnl_str,
                        "ROE": roe_str,
                        t("leverage"): f"{lev}x",
                        t("margin_used"): f"${margin_val:.2f}",
                        t("liq_price"): f"${liq_p:,.4f}" if liq_p else "-",
                        "TP1": tp1_val,
                        "TP2": tp2_val,
                        t("stop"): f"${pos.get('stop_price', 0):,.4f}",
                    }
                    f_pos_rows.append(row)
                st.dataframe(pd.DataFrame(f_pos_rows), use_container_width=True, hide_index=True)

                # --- Charts for Futures Open Positions ---
                st.markdown(f"""
                <div class="section-header">
                    <div class="section-icon" style="background:rgba(224,64,251,0.1);">
                        <span style="color:#E040FB;">C</span>
                    </div>
                    <div><p class="section-title">{t("candlestick_chart")} - {t("open_positions")} (Futures)</p></div>
                </div>
                """, unsafe_allow_html=True)
                f_chart_cols = st.columns(min(len(f_open_pos), 2))
                for idx, (sym, pos) in enumerate(f_open_pos.items()):
                    with f_chart_cols[idx % min(len(f_open_pos), 2)]:
                        pair = sym if "/" in sym else f"{sym}/USDT"
                        ohlcv_fpos = fetch_ohlcv(pair, "4h", limit=60)
                        if not ohlcv_fpos.empty:
                            fig_fpos = go.Figure(data=[go.Candlestick(
                                x=ohlcv_fpos["timestamp"],
                                open=ohlcv_fpos["open"], high=ohlcv_fpos["high"],
                                low=ohlcv_fpos["low"], close=ohlcv_fpos["close"],
                                increasing_line_color=BINANCE_GREEN, decreasing_line_color=BINANCE_RED,
                                increasing_fillcolor=BINANCE_GREEN, decreasing_fillcolor=BINANCE_RED,
                            )])
                            f_entry_p = pos.get("entry_price", 0)
                            f_stop_p = pos.get("stop_price", 0)
                            f_lev = pos.get("leverage", 3)
                            f_side_label = pos["side"].upper()
                            fig_fpos.add_hline(y=f_entry_p, line_dash="dash", line_color=BINANCE_YELLOW,
                                               annotation_text=f"Entry ${f_entry_p:,.2f}", annotation_font_color=BINANCE_YELLOW)
                            if f_stop_p > 0:
                                fig_fpos.add_hline(y=f_stop_p, line_dash="dash", line_color=BINANCE_RED,
                                                   annotation_text=f"Stop ${f_stop_p:,.2f}", annotation_font_color=BINANCE_RED)
                            if "tp1_price" in pos:
                                fig_fpos.add_hline(y=pos["tp1_price"], line_dash="dash", line_color=BINANCE_GREEN,
                                                   annotation_text=f"TP1 ${pos['tp1_price']:,.2f}", annotation_font_color=BINANCE_GREEN)
                            if "tp2_price" in pos:
                                fig_fpos.add_hline(y=pos["tp2_price"], line_dash="dot", line_color=BINANCE_GREEN,
                                                   annotation_text=f"TP2 ${pos['tp2_price']:,.2f}", annotation_font_color=BINANCE_GREEN)
                            if pos.get("liq_price", 0) > 0:
                                fig_fpos.add_hline(y=pos["liq_price"], line_dash="dash", line_color="#FF4444",
                                                   annotation_text=f"Liq ${pos['liq_price']:,.2f}", annotation_font_color="#FF4444")
                            f_current = ohlcv_fpos["close"].iloc[-1]
                            f_pnl_pct = ((f_current - f_entry_p) / f_entry_p * 100) if pos["side"] == "long" else ((f_entry_p - f_current) / f_entry_p * 100)
                            f_roe = f_pnl_pct * f_lev
                            f_pnl_color = BINANCE_GREEN if f_pnl_pct >= 0 else BINANCE_RED
                            fig_fpos.update_layout(
                                title=dict(text=f"{sym} [{f_side_label} {f_lev}x] | Now: ${f_current:,.2f} (PnL {f_pnl_pct:+.2f}% | ROE {f_roe:+.2f}%)",
                                           font=dict(color=f_pnl_color, size=14)),
                                height=350, margin=dict(l=0, r=0, t=35, b=0),
                                xaxis_rangeslider_visible=False,
                            )
                            apply_binance_theme(fig_fpos)
                            st.plotly_chart(fig_fpos, use_container_width=True)
                        else:
                            st.warning(f"Cannot load chart for {sym}")
            else:
                st.info(t("no_open_positions"))

            st.divider()

            # --- Futures Trade History ---
            st.markdown(f"##### {t('trade_history')}")
            if f_history:
                f_hist_rows = []
                for trade in f_history[-15:]:
                    roe_val = trade.get("roe", 0)
                    dur = trade.get("duration_hours")
                    if dur is not None:
                        dur_h = int(dur)
                        dur_m = int((dur - dur_h) * 60)
                        dur_str = f"{dur_h}h {dur_m}m"
                    else:
                        dur_str = "-"
                    f_hist_rows.append({
                        t("symbol"): trade["symbol"],
                        t("side"): trade["side"].upper(),
                        "Strategy": trade.get("strategy", "-"),
                        t("entry"): f"${trade['entry_price']:,.4f}",
                        t("exit"): f"${trade['exit_price']:,.4f}",
                        "PnL $": f"${trade['pnl_usd']:+.2f}",
                        "PnL %": f"{trade['pnl_pct']:+.2f}%",
                        t("roe"): f"{roe_val:+.2f}%",
                        t("duration"): dur_str,
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
                    marker=dict(size=8, color=BINANCE_YELLOW),
                    name=t("equity"),
                ))
                f_init_cap = f_eq_points[0]["equity"]
                fig_feq.add_hline(
                    y=f_init_cap, line_dash="dash", line_color="#848E9C",
                    annotation_text=f"{t('initial_capital')} ${f_init_cap:.0f}",
                    annotation_font_color="#848E9C",
                )
                feq_min = f_eq_df["equity"].min()
                feq_max = f_eq_df["equity"].max()
                feq_pad = max((feq_max - feq_min) * 0.2, 2)
                fig_feq.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("equity") + " ($)",
                    yaxis_range=[feq_min - feq_pad, feq_max + feq_pad],
                    hovermode="x unified",
                )
                apply_binance_theme(fig_feq)
                st.plotly_chart(fig_feq, use_container_width=True)
            else:
                st.info(t("equity_after_trade"))

            # --- Futures Per-Strategy Performance ---
            if f_history:
                st.divider()
                st.markdown(f"""
                <div class="section-header">
                    <div class="section-icon" style="background:rgba(224,64,251,0.1);">
                        <span style="color:#E040FB;">S</span>
                    </div>
                    <div><p class="section-title">{t("strategy_performance")} (Futures)</p></div>
                </div>
                """, unsafe_allow_html=True)

                f_strategy_map = {}
                for trade in f_history:
                    strat = trade.get("strategy", "Unknown")
                    if strat not in f_strategy_map:
                        f_strategy_map[strat] = []
                    f_strategy_map[strat].append(trade)

                f_strat_names = list(f_strategy_map.keys())
                f_cols_per_row = min(len(f_strat_names), 4)
                f_strat_cols = st.columns(f_cols_per_row) if f_cols_per_row > 0 else []

                for idx, (strat_name, trades) in enumerate(f_strategy_map.items()):
                    col = f_strat_cols[idx % f_cols_per_row]
                    n_trades = len(trades)
                    n_wins = sum(1 for t_ in trades if t_["pnl_usd"] > 0)
                    wr = (n_wins / n_trades * 100) if n_trades > 0 else 0
                    avg_pnl = sum(t_["pnl_usd"] for t_ in trades) / n_trades if n_trades > 0 else 0
                    total_pnl_strat = sum(t_["pnl_usd"] for t_ in trades)
                    avg_roe = sum(t_.get("roe", 0) for t_ in trades) / n_trades if n_trades > 0 else 0

                    header_color = BINANCE_GREEN if total_pnl_strat >= 0 else BINANCE_RED
                    pnl_sign = "+" if total_pnl_strat >= 0 else ""
                    avg_sign = "+" if avg_pnl >= 0 else ""
                    roe_sign = "+" if avg_roe >= 0 else ""

                    card_html = f"""
                    <div style="background-color:#1E2329;border:1px solid #2B3139;border-radius:8px;
                                padding:0;margin-bottom:8px;overflow:hidden;">
                        <div style="background-color:{header_color};padding:8px 12px;">
                            <span style="color:#181A20;font-weight:700;font-size:14px;">{strat_name}</span>
                        </div>
                        <div style="padding:10px 12px;font-size:13px;color:#EAECEF;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                <span style="color:#848E9C;">{t('total_trades')}</span>
                                <span>{n_trades}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                <span style="color:#848E9C;">{t('win_rate')}</span>
                                <span style="color:{BINANCE_GREEN if wr >= 50 else BINANCE_RED};">{wr:.1f}%</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                <span style="color:#848E9C;">{t('avg_pnl')}</span>
                                <span style="color:{BINANCE_GREEN if avg_pnl >= 0 else BINANCE_RED};">{avg_sign}${avg_pnl:.2f}</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                                <span style="color:#848E9C;">Avg ROE</span>
                                <span style="color:{BINANCE_GREEN if avg_roe >= 0 else BINANCE_RED};">{roe_sign}{avg_roe:.1f}%</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;">
                                <span style="color:#848E9C;">{t('total_return')}</span>
                                <span style="color:{BINANCE_GREEN if total_pnl_strat >= 0 else BINANCE_RED};">{pnl_sign}${total_pnl_strat:.2f}</span>
                            </div>
                        </div>
                    </div>
                    """
                    with col:
                        st.markdown(card_html, unsafe_allow_html=True)

        else:
            st.info(t("no_futures_data"))


# ═══════════════════════════════════════════════════════════════
# BACKTEST TAB
# ═══════════════════════════════════════════════════════════════

with tab_backtest:
    # Spot / Futures selector for backtest
    bt_mode_col, _ = st.columns([2, 5])
    with bt_mode_col:
        bt_mode = st.radio(
            "Backtest Mode",
            options=["spot", "futures"],
            format_func=lambda x: "Spot (Long-Only)" if x == "spot" else f"Futures (Long+Short, 3x)",
            horizontal=True,
            key="bt_mode_select",
            label_visibility="collapsed",
        )

    bt_df = load_backtest_results(mode=bt_mode)
    opt_df = load_optimized_results()
    eq_curves = load_equity_curves(mode=bt_mode)

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
            mode_label = "Spot (Long-Only)" if bt_mode == "spot" else "Futures (Long+Short, 3x)"
            st.markdown(f"#### {t('backtest_results')} — {mode_label}")
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
                color_palette = [
                    BINANCE_YELLOW, BINANCE_GREEN, "#3861FB", "#E040FB",
                    "#FF6D00", "#00E5FF", "#76FF03", "#FF1744",
                    "#FFAB00", "#D500F9", "#00B0FF", "#64DD17",
                ]

                # Get unique symbols for filter
                symbols_in_eq = sorted(set(
                    cd.get("symbol", curve_name.split("|")[0])
                    for curve_name, cd in eq_curves.items()
                    if isinstance(cd, dict) and "equity" in cd
                ))
                sel_sym_eq = st.selectbox(
                    t("select_coin"), [t("all")] + symbols_in_eq, key="eq_curve_sym",
                )

                # --- Equity curves chart ---
                fig_eq_bt = go.Figure()
                idx = 0
                for curve_name, curve_data in eq_curves.items():
                    if not isinstance(curve_data, dict) or "equity" not in curve_data:
                        continue
                    sym = curve_data.get("symbol", curve_name.split("|")[0])
                    if sel_sym_eq != t("all") and sym != sel_sym_eq:
                        continue
                    strat = curve_data.get("strategy", curve_name)
                    dates = curve_data["dates"]
                    equity = curve_data["equity"]
                    label = f"{sym} | {strat}"
                    fig_eq_bt.add_trace(go.Scatter(
                        x=dates, y=equity, mode="lines",
                        name=label,
                        line=dict(color=color_palette[idx % len(color_palette)], width=1.5),
                    ))
                    idx += 1

                # Buy & Hold reference (first matching curve)
                for curve_name, curve_data in eq_curves.items():
                    if not isinstance(curve_data, dict) or "buy_hold" not in curve_data:
                        continue
                    sym = curve_data.get("symbol", curve_name.split("|")[0])
                    if sel_sym_eq != t("all") and sym != sel_sym_eq:
                        continue
                    fig_eq_bt.add_trace(go.Scatter(
                        x=curve_data["dates"], y=curve_data["buy_hold"],
                        mode="lines", name=f"{sym} Buy & Hold",
                        line=dict(color="#848E9C", width=1, dash="dash"),
                    ))
                    break  # Only one buy & hold per symbol

                fig_eq_bt.add_hline(y=500, line_dash="dot", line_color="#2B3139")
                fig_eq_bt.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("equity") + " ($)",
                    hovermode="x unified",
                    legend=dict(bgcolor="rgba(30,35,41,0.8)", font=dict(color="#EAECEF")),
                )
                apply_binance_theme(fig_eq_bt)
                st.plotly_chart(fig_eq_bt, use_container_width=True)

                # --- Drawdown chart ---
                st.markdown(f"##### {t('drawdown')}")
                fig_dd = go.Figure()
                idx = 0
                for curve_name, curve_data in eq_curves.items():
                    if not isinstance(curve_data, dict) or "drawdown" not in curve_data:
                        continue
                    sym = curve_data.get("symbol", curve_name.split("|")[0])
                    if sel_sym_eq != t("all") and sym != sel_sym_eq:
                        continue
                    strat = curve_data.get("strategy", curve_name)
                    fig_dd.add_trace(go.Scatter(
                        x=curve_data["dates"], y=curve_data["drawdown"],
                        mode="lines", name=f"{sym} | {strat}",
                        line=dict(color=color_palette[idx % len(color_palette)], width=1),
                        fill="tozeroy", fillcolor=f"rgba(246,70,93,0.05)",
                    ))
                    idx += 1
                fig_dd.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title=t("drawdown") + " (%)",
                    hovermode="x unified",
                    legend=dict(bgcolor="rgba(30,35,41,0.8)", font=dict(color="#EAECEF")),
                )
                apply_binance_theme(fig_dd)
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.info(t("no_data"))

        # ─── Rolling Metrics ─────────────────────────────────
        with bt_sub_rolling:
            st.markdown(f"#### {t('rolling_metrics')}")
            # Rolling Sharpe from equity curves data
            has_rolling = any(
                isinstance(cd, dict) and "rolling_sharpe" in cd
                for cd in eq_curves.values()
            ) if eq_curves else False

            if has_rolling:
                color_palette_r = [
                    BINANCE_YELLOW, BINANCE_GREEN, "#3861FB", "#E040FB",
                    "#FF6D00", "#00E5FF", "#76FF03", "#FF1744",
                ]
                sel_sym_roll = st.selectbox(
                    t("select_coin"),
                    [t("all")] + sorted(set(
                        cd.get("symbol", k.split("|")[0])
                        for k, cd in eq_curves.items()
                        if isinstance(cd, dict) and "rolling_sharpe" in cd
                    )),
                    key="roll_sym",
                )

                # Rolling Sharpe
                st.markdown("##### Rolling Sharpe Ratio")
                fig_roll = go.Figure()
                idx = 0
                for curve_name, curve_data in eq_curves.items():
                    if not isinstance(curve_data, dict) or "rolling_sharpe" not in curve_data:
                        continue
                    sym = curve_data.get("symbol", curve_name.split("|")[0])
                    if sel_sym_roll != t("all") and sym != sel_sym_roll:
                        continue
                    strat = curve_data.get("strategy", curve_name)
                    fig_roll.add_trace(go.Scatter(
                        x=curve_data["dates"], y=curve_data["rolling_sharpe"],
                        mode="lines", name=f"{sym} | {strat}",
                        line=dict(color=color_palette_r[idx % len(color_palette_r)], width=1.5),
                    ))
                    idx += 1
                fig_roll.add_hline(y=0, line_dash="dash", line_color="#848E9C")
                fig_roll.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="Sharpe Ratio",
                    hovermode="x unified",
                    legend=dict(bgcolor="rgba(30,35,41,0.8)", font=dict(color="#EAECEF")),
                )
                apply_binance_theme(fig_roll)
                st.plotly_chart(fig_roll, use_container_width=True)

                # Cumulative PnL
                st.markdown("##### Cumulative PnL")
                fig_cpnl = go.Figure()
                idx = 0
                for curve_name, curve_data in eq_curves.items():
                    if not isinstance(curve_data, dict) or "cum_pnl" not in curve_data:
                        continue
                    sym = curve_data.get("symbol", curve_name.split("|")[0])
                    if sel_sym_roll != t("all") and sym != sel_sym_roll:
                        continue
                    strat = curve_data.get("strategy", curve_name)
                    fig_cpnl.add_trace(go.Scatter(
                        x=curve_data["dates"], y=curve_data["cum_pnl"],
                        mode="lines", name=f"{sym} | {strat}",
                        line=dict(color=color_palette_r[idx % len(color_palette_r)], width=1.5),
                    ))
                    idx += 1
                fig_cpnl.add_hline(y=0, line_dash="dash", line_color="#848E9C")
                fig_cpnl.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title="PnL ($)",
                    hovermode="x unified",
                    legend=dict(bgcolor="rgba(30,35,41,0.8)", font=dict(color="#EAECEF")),
                )
                apply_binance_theme(fig_cpnl)
                st.plotly_chart(fig_cpnl, use_container_width=True)
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
                    has_mtf = "mtf_confirmed" in signal_df.columns

                    # --- MTF Overview Card ---
                    if has_mtf:
                        st.markdown(f"###### {t('mtf_overview')}")
                        mtf_cols = st.columns(4)
                        mtf_confirmed_count = int(signal_df["mtf_confirmed"].sum()) if has_mtf else 0
                        mtf_rejected_count = int(signal_df["action"].eq("rejected_by_mtf").sum()) if "action" in signal_df.columns else 0
                        mtf_cols[0].metric(t("mtf_confirmed_count"), mtf_confirmed_count)
                        mtf_cols[1].metric(t("mtf_rejected_count"), mtf_rejected_count)

                        # Win rate comparison (MTF confirmed vs not)
                        if "pnl_usd" in signal_df.columns:
                            confirmed_trades = signal_df[signal_df["mtf_confirmed"] == True]
                            not_confirmed_trades = signal_df[(signal_df["mtf_confirmed"] == False) & (signal_df["action"] != "rejected_by_mtf")]
                            wr_confirmed = (confirmed_trades["pnl_usd"] > 0).mean() * 100 if len(confirmed_trades) > 0 else 0
                            wr_not_confirmed = (not_confirmed_trades["pnl_usd"] > 0).mean() * 100 if len(not_confirmed_trades) > 0 else 0
                            mtf_cols[2].metric(t("mtf_confirmed_wr"), f"{wr_confirmed:.0f}%")
                            mtf_cols[3].metric(t("mtf_not_confirmed_wr"), f"{wr_not_confirmed:.0f}%")
                        st.divider()

                    # Filters
                    if has_mtf:
                        filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
                    else:
                        filt_col1, filt_col2, filt_col3 = st.columns(3)
                        filt_col4 = None
                    with filt_col1:
                        actions = [t("all")] + sorted(signal_df["action"].unique().tolist()) if "action" in signal_df.columns else [t("all")]
                        sel_action = st.selectbox(t("filter_action"), actions, key="spot_sig_action")
                    with filt_col2:
                        symbols = [t("all")] + sorted(signal_df["symbol"].unique().tolist()) if "symbol" in signal_df.columns else [t("all")]
                        sel_symbol = st.selectbox(t("filter_symbol"), symbols, key="spot_sig_symbol")
                    with filt_col3:
                        strategies = [t("all")] + sorted(signal_df["strategy"].unique().tolist()) if "strategy" in signal_df.columns else [t("all")]
                        sel_strategy = st.selectbox(t("filter_strategy"), strategies, key="spot_sig_strategy")
                    if has_mtf and filt_col4 is not None:
                        with filt_col4:
                            mtf_options = [t("all"), "Confirmed", "Not Confirmed"]
                            sel_mtf = st.selectbox(t("mtf_filter"), mtf_options, key="spot_sig_mtf")
                    else:
                        sel_mtf = t("all")

                    filtered = signal_df.copy()
                    if sel_action != t("all") and "action" in filtered.columns:
                        filtered = filtered[filtered["action"] == sel_action]
                    if sel_symbol != t("all") and "symbol" in filtered.columns:
                        filtered = filtered[filtered["symbol"] == sel_symbol]
                    if sel_strategy != t("all") and "strategy" in filtered.columns:
                        filtered = filtered[filtered["strategy"] == sel_strategy]
                    if has_mtf and sel_mtf != t("all"):
                        if sel_mtf == "Confirmed":
                            filtered = filtered[filtered["mtf_confirmed"] == True]
                        else:
                            filtered = filtered[filtered["mtf_confirmed"] == False]

                    # Add MTF color-coded columns for display
                    display_df = filtered.tail(50).copy()
                    if has_mtf:
                        def _mtf_color(val):
                            if val == "bullish":
                                return f'<span style="color:{BINANCE_GREEN};font-weight:600">{val}</span>'
                            elif val == "bearish":
                                return f'<span style="color:{BINANCE_RED};font-weight:600">{val}</span>'
                            return f'<span style="color:#848E9C">{val}</span>'

                        def _mtf_check(val):
                            if val is True or val == True:
                                return f'<span style="color:{BINANCE_GREEN}">&#10003;</span>'
                            return f'<span style="color:{BINANCE_RED}">&#10007;</span>'

                        if "mtf_daily" in display_df.columns:
                            display_df[t("mtf_daily")] = display_df["mtf_daily"].apply(_mtf_color)
                        if "mtf_hourly" in display_df.columns:
                            display_df[t("mtf_hourly")] = display_df["mtf_hourly"].apply(_mtf_color)
                        if "mtf_confirmed" in display_df.columns:
                            display_df[t("mtf_confirmed")] = display_df["mtf_confirmed"].apply(_mtf_check)
                        # Drop raw columns if display columns were added
                        for col in ["mtf_daily", "mtf_hourly", "mtf_confirmed"]:
                            if col in display_df.columns and t(col) in display_df.columns and col != t(col):
                                display_df = display_df.drop(columns=[col])

                        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
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

                # --- PnL by Strategy (Futures) ---
                if "strategy" in f_pnl_data.columns:
                    st.divider()
                    st.markdown(f"##### {t('pnl_by_strategy')} (Futures)")
                    f_pnl_strat = f_pnl_data.groupby("strategy")["pnl_usd"].sum().sort_values(ascending=False)
                    fig_fstrat = go.Figure()
                    fstrat_colors = [BINANCE_GREEN if p > 0 else BINANCE_RED for p in f_pnl_strat.values]
                    fig_fstrat.add_trace(go.Bar(
                        x=f_pnl_strat.index,
                        y=f_pnl_strat.values,
                        marker_color=fstrat_colors,
                        text=[f"${v:+.2f}" for v in f_pnl_strat.values],
                        textposition="outside",
                        textfont=dict(color="#EAECEF"),
                    ))
                    fig_fstrat.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title="PnL ($)",
                    )
                    apply_binance_theme(fig_fstrat)
                    st.plotly_chart(fig_fstrat, use_container_width=True)

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
                display_f_cols = ["symbol", "side", "strategy", "entry_price", "exit_price",
                                  "pnl_usd", "pnl_pct", "roe", "duration_hours", "reason", "closed_at"]
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
        f"{t('last_updated')}: {datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M (VN)')}"
    )
with footer_right:
    st.caption(f"{t('auto_refresh')}: 5 min")

# Auto-refresh every 5 minutes (300 seconds)
st.markdown(
    """<meta http-equiv="refresh" content="300">""",
    unsafe_allow_html=True,
)
