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

from trading import manual_actions
from config.settings import MAX_POSITIONS

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

/* ===== TABS — top-level pill nav ===== */
div[data-testid="stTabs"] {
    background-color: transparent;
}
/* Top-level tab list: padding + subtle separator */
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] {
    background: #1E2329;
    border: 1px solid #2B3139;
    border-radius: 10px;
    padding: 6px;
    gap: 4px;
    margin-bottom: 12px;
    overflow-x: auto;
}
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
    color: #848E9C !important;
    background-color: transparent !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    border-radius: 7px !important;
    border: none !important;
    transition: all 0.15s ease !important;
    white-space: nowrap;
    margin: 0 !important;
}
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
    color: #EAECEF !important;
    background-color: rgba(252,213,53,0.06) !important;
}
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #0B0E11 !important;
    background: linear-gradient(135deg, #FCD535 0%, #F0B90B 100%) !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(240,185,11,0.25);
}
/* Hide the tab-list bottom indicator (we use full bg now) */
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ===== Nested (sub) tabs — keep slim underline style ===== */
div[data-testid="stTabs"] div[data-baseweb="tab-list"] div[data-baseweb="tab-list"] button[data-baseweb="tab"] {
    color: #848E9C !important;
    background-color: transparent !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
}
div[data-testid="stTabs"] div[data-baseweb="tab-list"] div[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #FCD535 !important;
    background: transparent !important;
    border-bottom: 2px solid #FCD535 !important;
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
        "tab_analysis": "Phân tích thị trường",
        "tab_signals": "Tín hiệu",
        "tab_trading": "Giao dịch",
        "tab_backtest": "Phân tích Backtest",
        "tab_performance": "Hiệu suất",
        "fng_label": "Chỉ số Sợ hãi & Tham lam",
        "btc_dom_label": "Thống trị của BTC",
        "btc_dom_sub": "của tổng market cap",
        "total_mcap_label": "Tổng Market Cap",
        "regime_hint_label": "Gợi ý ngày hôm nay",
        "regime_extreme_fear": "Sợ hãi cực độ — thường là vùng DCA / mua dần",
        "regime_fear": "Sợ hãi — thận trọng, ưu tiên chiến lược contrarian",
        "regime_neutral": "Trung lập — chiến lược bình thường",
        "regime_greed": "Tham lam — tránh đu đỉnh",
        "regime_extreme_greed": "Tham lam cực độ — nguy cơ đảo chiều cao",
        "top_gainers": "Top tăng mạnh 24h",
        "top_losers": "Top giảm mạnh 24h",
        "no_data_available": "Không có dữ liệu",
        "heatmap_title": "Heatmap watchlist 24h",
        "funding_title": "Funding Rate (Binance USDT-M)",
        "funding_low": "Thấp nhất (short trả cho long — áp lực tăng)",
        "funding_high": "Cao nhất (long trả cho short — long over-leverage)",
        "signals_live_title": "Tín hiệu trực tiếp từ bot",
        "signals_cache_hint": "cập nhật mỗi 5 phút",
        "signals_caption": "Quét chính xác logic strategy bot dùng. Khi bot tick tiếp theo các tín hiệu dưới đây sẽ được bot đánh giá.",
        "signals_scanning": "Đang quét tín hiệu…",
        "signals_no_data": "Chưa có dữ liệu tín hiệu (signal generator có thể đang khởi động hoặc API lỗi).",
        "signals_scanned": "Coin quét",
        "signals_active": "Có tín hiệu",
        "signals_neutral": "Trung lập",
        "signals_active_title": "Coin đang cho tín hiệu vào lệnh",
        "signals_none_active": "Hiện không có coin nào cho tín hiệu vào lệnh. Bot sẽ tiếp tục quan sát.",
        "signals_neutral_expander": "coin trung lập — đang quan sát, chờ điều kiện",
        "refresh_data": "Làm mới dữ liệu",
        "load_error": "Lỗi tải dữ liệu — bấm làm mới để thử lại",
        "perf_breakdown_title": "Phân tích chi tiết",
        "by_hour_vn": "PnL theo giờ (giờ VN)",
        "by_dayofweek": "PnL theo ngày trong tuần",
        "by_regime": "PnL theo regime thị trường",
        "daily_pnl_calendar": "Lịch PnL theo ngày",
        "drawdown_curve": "Đường cong drawdown",
        "best_hour": "Giờ thắng nhiều nhất",
        "worst_hour": "Giờ thua nhiều nhất",
        "best_day": "Ngày tốt nhất",
        "worst_day": "Ngày xấu nhất",
        "guard_status": "Trạng thái phòng vệ",
        "daily_loss_so_far": "Lỗ trong ngày (UTC)",
        "blacklisted_coins": "Coin tạm cấm",
        "no_blacklist": "Không có coin nào bị cấm",
        "today_at_glance": "Hôm nay",
        "today_trades": "Lệnh hôm nay",
        "today_pnl": "PnL hôm nay",
        "today_no_trades": "Chưa có lệnh nào hôm nay",
        "trades_short": "lệnh",
        "control_panel": "Bảng điều khiển",
        "bot_status": "Trạng thái bot",
        "status_active": "Hoạt động",
        "status_paused": "Tạm dừng",
        "status_killed": "Kill switch",
        "pause_for": "Tạm dừng trong",
        "pause_btn": "⏸️ Tạm dừng",
        "resume_btn": "▶️ Hoạt động lại",
        "pause_queued": "Đã queue lệnh tạm dừng (≤30s)",
        "resume_queued": "Đã queue lệnh resume (≤30s)",
        "to_tp1": "đến TP1",
        "to_tp2": "đến TP2",
        "to_stop": "đến SL",
        "held_for": "Thời gian giữ",
        "close_pct": "Đóng {pct}%",
        "close_all": "Đóng 100%",
        "close_queued": "⏳ Đã queue đóng {pct}% {sym} (≤30s)",
        "pending_n": "⏳ {n} hành động chờ — bot xử lý trong ≤30s",
        "manual_close_section": "🔥 Đóng vị thế thủ công",
        "progress_help": "Vị trí giá hiện tại trên thanh SL → Entry → TP1 → TP2",
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
        "tab_analysis": "Market Analysis",
        "tab_signals": "Signals",
        "tab_trading": "Trading",
        "tab_backtest": "Backtest Analysis",
        "tab_performance": "Performance",
        "fng_label": "Fear & Greed",
        "btc_dom_label": "BTC Dominance",
        "btc_dom_sub": "of total market cap",
        "total_mcap_label": "Total Market Cap",
        "regime_hint_label": "Today's hint",
        "regime_extreme_fear": "Extreme fear — usually a DCA / dip-buy zone",
        "regime_fear": "Fear — be cautious, favor contrarian strategies",
        "regime_neutral": "Neutral — standard playbook",
        "regime_greed": "Greed — avoid chasing tops",
        "regime_extreme_greed": "Extreme greed — high reversal risk",
        "top_gainers": "Top Gainers 24h",
        "top_losers": "Top Losers 24h",
        "no_data_available": "No data",
        "heatmap_title": "Watchlist 24h heatmap",
        "funding_title": "Funding Rate (Binance USDT-M)",
        "funding_low": "Lowest (shorts paying longs — bullish pressure)",
        "funding_high": "Highest (longs paying shorts — overleveraged longs)",
        "signals_live_title": "Live signals from the bot",
        "signals_cache_hint": "refreshed every 5 min",
        "signals_caption": "Runs the same strategy the bot uses. The signals below will be evaluated by the bot on its next tick.",
        "signals_scanning": "Scanning signals…",
        "signals_no_data": "No signal data yet (signal generator may be starting up or API failed).",
        "signals_scanned": "Scanned",
        "signals_active": "Active",
        "signals_neutral": "Neutral",
        "signals_active_title": "Coins currently giving entry signals",
        "signals_none_active": "No coins currently signaling. The bot will keep watching.",
        "signals_neutral_expander": "neutral coins — being watched, waiting for conditions",
        "refresh_data": "Refresh data",
        "load_error": "Failed to load data — click refresh to retry",
        "perf_breakdown_title": "Detailed breakdown",
        "by_hour_vn": "PnL by hour (VN time)",
        "by_dayofweek": "PnL by day of week",
        "by_regime": "PnL by market regime",
        "daily_pnl_calendar": "Daily PnL calendar",
        "drawdown_curve": "Drawdown curve",
        "best_hour": "Best hour",
        "worst_hour": "Worst hour",
        "best_day": "Best day",
        "worst_day": "Worst day",
        "guard_status": "Guard status",
        "daily_loss_so_far": "Daily loss (UTC)",
        "blacklisted_coins": "Blacklisted coins",
        "no_blacklist": "No coins blacklisted",
        "today_at_glance": "Today",
        "today_trades": "Today's trades",
        "today_pnl": "Today's PnL",
        "today_no_trades": "No trades today yet",
        "trades_short": "trades",
        "control_panel": "Control panel",
        "bot_status": "Bot status",
        "status_active": "Active",
        "status_paused": "Paused",
        "status_killed": "Kill switch",
        "pause_for": "Pause for",
        "pause_btn": "⏸️ Pause",
        "resume_btn": "▶️ Resume",
        "pause_queued": "Pause action queued (≤30s)",
        "resume_queued": "Resume action queued (≤30s)",
        "to_tp1": "to TP1",
        "to_tp2": "to TP2",
        "to_stop": "to SL",
        "held_for": "Held for",
        "close_pct": "Close {pct}%",
        "close_all": "Close 100%",
        "close_queued": "⏳ Queued: close {pct}% {sym} (≤30s)",
        "pending_n": "⏳ {n} pending action(s) — bot processes within ≤30s",
        "manual_close_section": "🔥 Manual close",
        "progress_help": "Current price position on the SL → Entry → TP1 → TP2 strip",
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


@st.cache_data(ttl=300)
def fetch_ohlcv_batch(symbols: tuple, timeframe: str = "1h", limit: int = 48) -> dict:
    """Fetch OHLCV for many symbols in parallel. Returns {symbol: DataFrame}.
    Cached as a single entry so a cold load fans out once instead of N times.
    """
    if ccxt is None:
        return {symbol: pd.DataFrame() for symbol in symbols}
    from concurrent.futures import ThreadPoolExecutor

    def _one(sym):
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return sym, df
        except Exception:
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_one, symbols))
    return dict(results)


# ═══════════════════════════════════════════════════════════════
# MARKET ANALYSIS / SIGNALS FETCHERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_top_movers(top_n: int = 10) -> dict:
    """Top gainers & losers across all USDT spot pairs in the last 24h.
    Retries once on failure. Raises if both attempts fail (Streamlit
    won't cache the exception so the next call retries fresh).
    """
    if ccxt is None:
        return {"gainers": [], "losers": []}
    last_err = None
    for attempt in range(2):
        try:
            ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
            tickers = ex.fetch_tickers()
            break
        except Exception as e:
            last_err = e
            tickers = None
    if tickers is None:
        raise last_err or RuntimeError("fetch_tickers returned no data")

    rows = []
    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        pct = tk.get("percentage")
        last = tk.get("last")
        quote_vol = tk.get("quoteVolume") or 0
        if pct is None or last is None or quote_vol < 1_000_000:
            continue
        rows.append({
            "symbol": sym,
            "last": last,
            "pct_24h": pct,
            "quote_vol": quote_vol,
        })
    if not rows:
        raise RuntimeError("No liquid USDT pairs in ticker response")
    rows.sort(key=lambda r: r["pct_24h"], reverse=True)
    return {
        "gainers": rows[:top_n],
        "losers": list(reversed(rows[-top_n:])),
    }


@st.cache_data(ttl=300)
def fetch_funding_rates_batch(symbols: tuple) -> dict:
    """Fetch funding rate for a list of symbols via the strategies helper.
    Returns {symbol: {rate, avg_24h, trend}} or {} per failure.
    """
    try:
        from strategies.funding_rate import FundingRateStrategy
    except Exception:
        return {}
    from concurrent.futures import ThreadPoolExecutor

    fr = FundingRateStrategy()

    def _one(sym):
        try:
            return sym, fr.fetch_funding_rate(sym)
        except Exception:
            return sym, {}

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_one, symbols))
    return dict(results)


@st.cache_data(ttl=3600)
def fetch_fear_greed() -> dict:
    """Fear & Greed index (alternative.me). Returns {value, classification} or {}."""
    try:
        import requests
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json().get("data", [])
        if not data:
            return {}
        return {
            "value": int(data[0]["value"]),
            "classification": data[0]["value_classification"],
        }
    except Exception:
        return {}


@st.cache_data(ttl=900)
def fetch_btc_dominance() -> dict:
    """BTC dominance via CoinGecko global. Returns {btc_dom, total_mcap_usd, mcap_change_24h} or {}."""
    try:
        import requests
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        d = r.json().get("data", {})
        if not d:
            return {}
        return {
            "btc_dom": d.get("market_cap_percentage", {}).get("btc", 0),
            "total_mcap_usd": d.get("total_market_cap", {}).get("usd", 0),
            "mcap_change_24h": d.get("market_cap_change_percentage_24h_usd", 0),
        }
    except Exception:
        return {}


@st.cache_resource
def _get_signal_generator():
    """SignalGenerator is expensive to construct (loads ML model, strategies).
    Cache as resource so it's built once per Streamlit session.
    """
    from trading.signal_generator import SignalGenerator
    return SignalGenerator()


@st.cache_data(ttl=300)
def fetch_live_signals() -> list:
    """Run the bot's own SignalGenerator across all configured coins.
    Returns a list of signal dicts (same shape as signal_generator outputs).
    """
    try:
        sg = _get_signal_generator()
        return sg.generate_all()
    except Exception:
        return []


def render_advanced_breakdowns(history: list, key_prefix: str = ""):
    """Render performance breakdowns shared across spot/futures perf tabs.
    Sections: by hour-of-day (VN), by weekday, by regime, daily PnL calendar,
    drawdown curve.
    """
    if not history:
        return
    df = pd.DataFrame(history)
    if "closed_at" not in df.columns or "pnl_usd" not in df.columns:
        return
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["closed_at"])
    if df.empty:
        return
    df["closed_vn"] = df["closed_at"].dt.tz_convert(VN_TZ)
    df["hour_vn"] = df["closed_vn"].dt.hour
    df["weekday"] = df["closed_vn"].dt.day_name()
    df["date_vn"] = df["closed_vn"].dt.date

    st.markdown(f"#### 🔬 {t('perf_breakdown_title')}")

    # ── Hour of day + weekday side by side ──
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown(f"##### {t('by_hour_vn')}")
        hour_pnl = df.groupby("hour_vn")["pnl_usd"].sum().reindex(range(24), fill_value=0)
        colors = [BINANCE_GREEN if v >= 0 else BINANCE_RED for v in hour_pnl.values]
        fig_h = go.Figure(go.Bar(x=hour_pnl.index, y=hour_pnl.values, marker_color=colors))
        fig_h.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Hour (VN)", yaxis_title="PnL ($)",
            xaxis=dict(tickmode="linear", dtick=2),
        )
        apply_binance_theme(fig_h)
        st.plotly_chart(fig_h, use_container_width=True, key=f"{key_prefix}_hour")
        if hour_pnl.abs().sum() > 0:
            best_h = hour_pnl.idxmax()
            worst_h = hour_pnl.idxmin()
            st.caption(f"🏆 {t('best_hour')}: **{best_h:02d}h** (${hour_pnl[best_h]:+.2f}) "
                       f"· 💀 {t('worst_hour')}: **{worst_h:02d}h** (${hour_pnl[worst_h]:+.2f})")

    with bcol2:
        st.markdown(f"##### {t('by_dayofweek')}")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        labels = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"] if st.session_state.lang == "vi" \
                 else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wd_pnl = df.groupby("weekday")["pnl_usd"].sum().reindex(order, fill_value=0)
        colors = [BINANCE_GREEN if v >= 0 else BINANCE_RED for v in wd_pnl.values]
        fig_w = go.Figure(go.Bar(x=labels, y=wd_pnl.values, marker_color=colors,
                                 text=[f"${v:+.1f}" for v in wd_pnl.values],
                                 textposition="outside",
                                 textfont=dict(color="#EAECEF")))
        fig_w.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="PnL ($)",
        )
        apply_binance_theme(fig_w)
        st.plotly_chart(fig_w, use_container_width=True, key=f"{key_prefix}_wd")
        if wd_pnl.abs().sum() > 0:
            best_d = wd_pnl.idxmax()
            worst_d = wd_pnl.idxmin()
            best_l = labels[order.index(best_d)]
            worst_l = labels[order.index(worst_d)]
            st.caption(f"🏆 {t('best_day')}: **{best_l}** (${wd_pnl[best_d]:+.2f}) "
                       f"· 💀 {t('worst_day')}: **{worst_l}** (${wd_pnl[worst_d]:+.2f})")

    # ── Regime breakdown ──
    if "regime" in df.columns and df["regime"].notna().any():
        st.markdown(f"##### {t('by_regime')}")
        rg_stats = df.groupby("regime").agg(
            trades=("pnl_usd", "count"),
            total_pnl=("pnl_usd", "sum"),
            win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
            avg_pnl=("pnl_usd", "mean"),
        ).round(2)
        rg_stats.columns = [t("total_trades"), "Total PnL ($)", f"{t('win_rate')} (%)", "Avg PnL ($)"]
        st.dataframe(rg_stats, use_container_width=True)

    # ── Daily PnL calendar (last 60 days) ──
    st.markdown(f"##### {t('daily_pnl_calendar')}")
    daily = df.groupby("date_vn")["pnl_usd"].sum().sort_index()
    if len(daily) > 0:
        last_date = daily.index.max()
        first_date = max(daily.index.min(), last_date - timedelta(days=60))
        full_range = pd.date_range(first_date, last_date).date
        daily = daily.reindex(full_range, fill_value=0)
        rows = [list(daily.index[i:i + 7]) for i in range(0, len(daily), 7)]
        for week in rows:
            wcols = st.columns(7)
            for i, day in enumerate(week):
                pnl = daily.get(day, 0)
                if pnl > 5: bg = "#0ECB81"
                elif pnl > 0: bg = "rgba(14,203,129,0.4)"
                elif pnl == 0: bg = "rgba(132,142,156,0.15)"
                elif pnl > -5: bg = "rgba(246,70,93,0.4)"
                else: bg = "#F6465D"
                txt_color = "#FFF" if abs(pnl) > 5 else "#EAECEF"
                wcols[i].markdown(
                    f"""<div style="padding:6px;background:{bg};border-radius:4px;text-align:center;min-height:50px;">
                    <div style="color:{txt_color};font-size:10px;">{day.strftime('%d/%m')}</div>
                    <div style="color:{txt_color};font-size:11px;font-weight:600;">${pnl:+.1f}</div>
                    </div>""", unsafe_allow_html=True,
                )

    # ── Drawdown curve (running) ──
    st.markdown(f"##### {t('drawdown_curve')}")
    df_sorted = df.sort_values("closed_at").reset_index(drop=True)
    df_sorted["cum_pnl"] = df_sorted["pnl_usd"].cumsum()
    df_sorted["peak"] = df_sorted["cum_pnl"].cummax()
    df_sorted["dd"] = df_sorted["cum_pnl"] - df_sorted["peak"]
    fig_dd = go.Figure(go.Scatter(
        x=df_sorted["closed_at"], y=df_sorted["dd"],
        fill="tozeroy", mode="lines",
        line=dict(color="#F6465D", width=1),
        fillcolor="rgba(246,70,93,0.3)",
    ))
    fig_dd.update_layout(
        height=240, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Drawdown ($)", xaxis_title=None,
    )
    apply_binance_theme(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True, key=f"{key_prefix}_dd")


def render_today_summary(spot_state: dict, fut_state: dict):
    """Hero 'Today at a Glance' card. Shows today's trade count, PnL split by
    spot/futures, win/loss, and any active warnings (pause, blacklist, kill).
    """
    today_utc = datetime.now(timezone.utc).date()

    def _today_trades(state):
        if not state:
            return [], 0.0, 0, 0
        hist = state.get("trade_history", [])
        today_trades = []
        for t_ in hist:
            ca = t_.get("closed_at")
            if not ca:
                continue
            try:
                d = datetime.fromisoformat(ca).date()
            except Exception:
                continue
            if d == today_utc:
                today_trades.append(t_)
        pnl = sum(t_.get("pnl_usd", 0) for t_ in today_trades)
        wins = sum(1 for t_ in today_trades if t_.get("pnl_usd", 0) > 0)
        losses = sum(1 for t_ in today_trades if t_.get("pnl_usd", 0) <= 0)
        return today_trades, pnl, wins, losses

    spot_trades, spot_pnl, spot_w, spot_l = _today_trades(spot_state)
    fut_trades, fut_pnl, fut_w, fut_l = _today_trades(fut_state)
    total_trades = len(spot_trades) + len(fut_trades)
    total_pnl = spot_pnl + fut_pnl
    total_w = spot_w + fut_w
    total_l = spot_l + fut_l

    # Warnings
    warnings = []
    for label, state in [("Spot", spot_state), ("Futures", fut_state)]:
        if not state:
            continue
        pu = state.get("paused_until")
        if pu:
            try:
                pu_dt = datetime.fromisoformat(pu)
                if pu_dt > datetime.now(timezone.utc):
                    hrs = (pu_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    warnings.append(f"⏸️ <b>{label}</b> paused ({hrs:.1f}h left)")
            except Exception:
                pass
        bl = state.get("symbol_blacklist", {})
        active_bl = []
        for sym, until in bl.items():
            try:
                until_ts = datetime.fromisoformat(until)
                if until_ts > datetime.now(timezone.utc):
                    active_bl.append(sym)
            except Exception:
                pass
        if active_bl:
            warnings.append(f"⛔ <b>{label}</b> blacklist: {', '.join(active_bl)}")

    def _fmt_money(v):
        sign = "+" if v >= 0 else "−"
        return f"{sign}${abs(v):,.2f}"

    pnl_color = BINANCE_GREEN if total_pnl >= 0 else BINANCE_RED
    win_rate = (total_w / total_trades * 100) if total_trades > 0 else 0
    capital_total = (spot_state.get("capital", 0) if spot_state else 0) + \
                    (fut_state.get("capital", 0) if fut_state else 0)
    pnl_pct = (total_pnl / capital_total * 100) if capital_total > 0 else 0

    def _card(label, value, value_color="#EAECEF", subtitle=""):
        sub_html = f'<div style="color:#5E6673;font-size:11px;margin-top:2px;">{subtitle}</div>' if subtitle else ""
        return f"""<div style="background:#0B0E11;border:1px solid #2B3139;border-radius:8px;
            padding:14px 16px;height:100%;">
            <div style="color:#848E9C;font-size:10px;letter-spacing:0.5px;text-transform:uppercase;
                font-weight:600;margin-bottom:6px;">{label}</div>
            <div style="color:{value_color};font-size:24px;font-weight:700;line-height:1.1;">{value}</div>
            {sub_html}
        </div>"""

    if total_trades == 0:
        body = f"""<div style="display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px;">
            <div style="background:#0B0E11;border:1px solid #2B3139;border-radius:8px;
                padding:20px;text-align:center;color:#5E6673;font-size:13px;">
                {t('today_no_trades')}
            </div>
        </div>"""
    else:
        # Spot card: empty if no spot trades, otherwise PnL + count
        if len(spot_trades):
            sc_color = BINANCE_GREEN if spot_pnl >= 0 else BINANCE_RED
            spot_card = _card(
                "·[spot]",
                _fmt_money(spot_pnl),
                value_color=sc_color,
                subtitle=f"{len(spot_trades)} {t('trades_short')} · {spot_w}W / {spot_l}L",
            )
        else:
            spot_card = _card("·[spot]", "—", value_color="#5E6673", subtitle=f"0 {t('trades_short')}")

        if len(fut_trades):
            fc_color = BINANCE_GREEN if fut_pnl >= 0 else BINANCE_RED
            fut_card = _card(
                "🔥[fut]",
                _fmt_money(fut_pnl),
                value_color=fc_color,
                subtitle=f"{len(fut_trades)} {t('trades_short')} · {fut_w}W / {fut_l}L",
            )
        else:
            fut_card = _card("🔥[fut]", "—", value_color="#5E6673", subtitle=f"0 {t('trades_short')}")

        body = f"""<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:10px;margin-top:10px;">
            {_card(t('today_pnl'), _fmt_money(total_pnl), value_color=pnl_color,
                   subtitle=f"{pnl_pct:+.2f}% capital")}
            {_card(t('today_trades'), str(total_trades),
                   subtitle=f'<span style="color:{BINANCE_GREEN}">●{total_w}W</span> '
                            f'<span style="color:{BINANCE_RED}">●{total_l}L</span> · '
                            f'WR {win_rate:.0f}%')}
            {spot_card}
            {fut_card}
        </div>"""

    warn_html = ""
    if warnings:
        warn_html = f"""<div style="margin-top:10px;padding:10px 14px;background:rgba(246,70,93,0.08);
        border-left:3px solid #F6465D;border-radius:4px;color:#EAECEF;font-size:12px;line-height:1.6;">
        {'<br>'.join(warnings)}
        </div>"""

    st.markdown(f"""
    <div style="background:#1E2329;border:1px solid #2B3139;border-radius:10px;
        padding:14px 18px;margin-bottom:16px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="color:#848E9C;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                font-weight:600;">
                📅 {t('today_at_glance')}
            </div>
            <div style="color:#5E6673;font-size:11px;">
                {datetime.now(VN_TZ).strftime('%A · %d/%m/%Y · %H:%M')} VN
            </div>
        </div>
        {body}
        {warn_html}
    </div>
    """, unsafe_allow_html=True)


def render_position_progress_bar(entry: float, current: float, stop: float,
                                 tp1: float, tp2: float, side: str) -> str:
    """Build an HTML progress bar showing position location between SL→entry→TP1→TP2."""
    if side == "long":
        lo = min(stop, entry, current) * 0.998
        hi = max(tp2, current, entry) * 1.002
    else:
        lo = min(tp2, current, entry) * 0.998
        hi = max(stop, entry, current) * 1.002

    span = hi - lo if hi > lo else 1
    def _pos_pct(p): return max(0, min(100, (p - lo) / span * 100))

    entry_pct = _pos_pct(entry)
    cur_pct = _pos_pct(current)
    tp1_pct = _pos_pct(tp1)
    tp2_pct = _pos_pct(tp2)
    sl_pct = _pos_pct(stop)

    # Profit zone color depends on side and current vs entry
    in_profit = (current > entry) if side == "long" else (current < entry)
    cur_color = BINANCE_GREEN if in_profit else BINANCE_RED

    # Single-line HTML — multi-line indented HTML inside st.markdown gets
    # parsed as a markdown code block, leaking the raw tags as text.
    return (
        f'<div style="position:relative;height:32px;background:linear-gradient(to right,'
        f'rgba(246,70,93,0.15) 0%,rgba(246,70,93,0.15) {entry_pct}%,'
        f'rgba(14,203,129,0.15) {entry_pct}%,rgba(14,203,129,0.15) 100%);'
        f'border-radius:4px;margin:6px 0;">'
        f'<div style="position:absolute;left:{sl_pct}%;top:0;height:100%;width:2px;background:#F6465D;"></div>'
        f'<div style="position:absolute;left:{entry_pct}%;top:0;height:100%;width:2px;background:#848E9C;"></div>'
        f'<div style="position:absolute;left:{tp1_pct}%;top:0;height:100%;width:2px;background:#FCD535;"></div>'
        f'<div style="position:absolute;left:{tp2_pct}%;top:0;height:100%;width:2px;background:#0ECB81;"></div>'
        f'<div style="position:absolute;left:calc({cur_pct}% - 6px);top:6px;width:12px;height:20px;'
        f'background:{cur_color};border-radius:3px;border:2px solid #0B0E11;'
        f'box-shadow:0 0 0 1px {cur_color};"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#848E9C;'
        f'margin-top:-2px;margin-bottom:6px;">'
        f'<span style="color:#F6465D;">SL ${stop:,.4f}</span>'
        f'<span>Entry ${entry:,.4f}</span>'
        f'<span style="color:#FCD535;">TP1 ${tp1:,.4f}</span>'
        f'<span style="color:#0ECB81;">TP2 ${tp2:,.4f}</span>'
        f'</div>'
    )


def render_sidebar(spot_state: dict, fut_state: dict):
    """Persistent left-side control panel: bot status, pause toggle, refresh."""
    with st.sidebar:
        st.markdown(f"### 🎛️ {t('control_panel')}")

        # ── Status indicators ──
        for mode_label, state, mode_key in [("·[spot]", spot_state, "spot"),
                                            ("🔥[fut]", fut_state, "futures")]:
            if not state:
                st.caption(f"{mode_label} (no state)")
                continue
            status_emoji = "🟢"
            status_label = t("status_active")
            status_color = "#0ECB81"
            pu = state.get("paused_until")
            if pu:
                try:
                    pu_dt = datetime.fromisoformat(pu)
                    now = datetime.now(timezone.utc)
                    if pu_dt > now:
                        hrs = (pu_dt - now).total_seconds() / 3600
                        status_emoji = "⏸️"
                        status_label = f"{t('status_paused')} ({hrs:.1f}h)"
                        status_color = "#FCD535"
                except Exception:
                    pass

            cap = state.get("capital", 0)
            peak = state.get("peak_capital", cap or 1)
            dd = (cap - peak) / peak if peak > 0 else 0
            from config.settings import MAX_DRAWDOWN as _MDD
            if dd < -_MDD:
                status_emoji = "🔴"
                status_label = t("status_killed")
                status_color = "#F6465D"

            st.markdown(
                f"""<div style="padding:8px 10px;background:#1E2329;border-radius:6px;margin-bottom:6px;">
                <div style="font-size:11px;color:#848E9C;">{mode_label}</div>
                <div style="font-size:14px;color:{status_color};font-weight:700;">{status_emoji} {status_label}</div>
                </div>""", unsafe_allow_html=True,
            )

            paused_now = pu and datetime.fromisoformat(pu) > datetime.now(timezone.utc)
            if paused_now:
                if st.button(t("resume_btn"), key=f"sb_resume_{mode_key}", use_container_width=True):
                    manual_actions.append_resume(mode_key)
                    st.success(t("resume_queued"))
                    st.rerun()
            else:
                pcol1, pcol2 = st.columns([2, 1])
                hours = pcol1.selectbox(
                    "Pause hours", [1, 4, 8, 24], index=1,
                    key=f"sb_pause_h_{mode_key}", label_visibility="collapsed",
                )
                if pcol2.button("⏸️", key=f"sb_pause_{mode_key}", use_container_width=True,
                                help=f"{t('pause_for')} {hours}h"):
                    manual_actions.append_pause(mode_key, hours)
                    st.success(t("pause_queued"))
                    st.rerun()
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 " + t("refresh_data"), use_container_width=True, key="sb_refresh_all"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"⏰ {datetime.now(VN_TZ).strftime('%H:%M (VN)')}")
        st.caption(f"📊 Auto-refresh: 15min")


def render_guard_status(state: dict):
    """Show active safety guards (daily loss, blacklisted coins, pause status)."""
    if not state:
        return
    st.markdown(f"##### 🛡️ {t('guard_status')}")
    gc1, gc2, gc3 = st.columns(3)

    daily_loss = state.get("daily_loss_pnl", 0.0)
    capital = state.get("capital", 1)
    pct = (daily_loss / capital * 100) if capital else 0
    color = BINANCE_RED if daily_loss < 0 else BINANCE_GREEN
    gc1.markdown(
        f"""<div style="padding:10px;background:#1E2329;border-radius:6px;">
        <div style="color:#848E9C;font-size:11px;">{t('daily_loss_so_far')}</div>
        <div style="color:{color};font-size:18px;font-weight:700;">${daily_loss:+.2f}</div>
        <div style="color:#848E9C;font-size:11px;">{pct:+.2f}% (limit -3%)</div>
        </div>""", unsafe_allow_html=True,
    )

    streak = state.get("consecutive_losses", 0)
    streak_color = BINANCE_RED if streak >= 3 else (BINANCE_YELLOW if streak >= 2 else "#848E9C")
    gc2.markdown(
        f"""<div style="padding:10px;background:#1E2329;border-radius:6px;">
        <div style="color:#848E9C;font-size:11px;">{t('consec_losses')}</div>
        <div style="color:{streak_color};font-size:18px;font-weight:700;">{streak}</div>
        <div style="color:#848E9C;font-size:11px;">soft pause @ 4 · hard @ 5</div>
        </div>""", unsafe_allow_html=True,
    )

    paused_until = state.get("paused_until")
    if paused_until:
        try:
            pu = datetime.fromisoformat(paused_until)
            now = datetime.now(timezone.utc)
            if pu > now:
                remaining = pu - now
                hrs = remaining.total_seconds() / 3600
                gc3.markdown(
                    f"""<div style="padding:10px;background:#3D1414;border-radius:6px;">
                    <div style="color:#848E9C;font-size:11px;">⏸️ Paused</div>
                    <div style="color:#F6465D;font-size:18px;font-weight:700;">{hrs:.1f}h</div>
                    <div style="color:#848E9C;font-size:11px;">until {pu.astimezone(VN_TZ).strftime('%d/%m %H:%M')}</div>
                    </div>""", unsafe_allow_html=True,
                )
            else:
                gc3.success("Active")
        except Exception:
            gc3.success("Active")
    else:
        gc3.success("Active")

    # Blacklist
    bl = state.get("symbol_blacklist", {})
    now = datetime.now(timezone.utc)
    active_bl = []
    for sym, until in bl.items():
        try:
            until_ts = datetime.fromisoformat(until)
            if until_ts > now:
                hrs = (until_ts - now).total_seconds() / 3600
                active_bl.append(f"**{sym}** ({hrs:.1f}h)")
        except Exception:
            pass
    if active_bl:
        st.warning(f"⛔ {t('blacklisted_coins')}: {' · '.join(active_bl)}")
    else:
        st.caption(f"✅ {t('no_blacklist')}")


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

# Live tickers used for MTM equity. Cached 15min (matches dashboard refresh).
_tickers = fetch_tickers()


def _mark_price(sym: str, fallback: float) -> float:
    t = _tickers.get(sym) if _tickers else None
    if t and t.get("last"):
        return t["last"]
    return fallback


def _spot_pos_mtm(pos: dict) -> float:
    """Current market value of a spot position (size_base * mark)."""
    mark = _mark_price(pos.get("symbol", ""), pos.get("entry_price", 0))
    return pos.get("size_base", 0) * mark


def _fut_pos_mtm(pos: dict) -> float:
    """Margin + unrealized PnL on the notional."""
    margin = pos.get("margin", pos.get("size_usdt", 0) / pos.get("leverage", 3))
    entry = pos.get("entry_price", 0)
    mark = _mark_price(pos.get("symbol", ""), entry)
    size_base = pos.get("size_base", 0)
    if pos.get("side") == "short":
        upnl = (entry - mark) * size_base
    else:
        upnl = (mark - entry) * size_base
    return margin + upnl


# Load Spot state
_spot_state = load_state()
_spot_initial = _spot_state.get("initial_capital", 500)
_spot_equity = _spot_state["capital"]
for _sym, _p in _spot_state.get("open_positions", {}).items():
    _p_full = {**_p, "symbol": _p.get("symbol", _sym)}
    _spot_equity += _spot_pos_mtm(_p_full)
_spot_pnl = _spot_equity - _spot_initial
_spot_trades = _spot_state.get("total_trades", 0)
_spot_wins = _spot_state.get("total_wins", 0)
_spot_positions = len(_spot_state.get("open_positions", {}))

# Load Futures state
_fut_state = load_futures_state()
if _fut_state:
    _fut_initial = _fut_state.get("initial_capital", 500)
    _fut_equity = _fut_state.get("capital", 0)
    for _sym, _fp in _fut_state.get("open_positions", {}).items():
        _fp_full = {**_fp, "symbol": _fp.get("symbol", _sym)}
        _fut_equity += _fut_pos_mtm(_fp_full)
    _fut_pnl = _fut_equity - _fut_initial
    _fut_trades = _fut_state.get("total_trades", 0)
    _fut_wins = _fut_state.get("total_wins", 0)
    _fut_positions = len(_fut_state.get("open_positions", {}))
    _fut_leverage = 3
    for _fp in _fut_state.get("open_positions", {}).values():
        _fut_leverage = _fp.get("leverage", 3)
        break
else:
    _fut_initial = 0
    _fut_equity = 0
    _fut_pnl = 0
    _fut_trades = 0
    _fut_wins = 0
    _fut_positions = 0
    _fut_leverage = 3

# Totals
_total_initial = _spot_initial + _fut_initial
_total_equity = _spot_equity + _fut_equity
_total_pnl = _total_equity - _total_initial
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
    _hero_max_pos = MAX_POSITIONS
    _hero_badge = "SPOT"
elif _view_mode == "Futures":
    _hero_equity = _fut_equity
    _hero_pnl = _fut_pnl
    _hero_trades = _fut_trades
    _hero_wins = _fut_wins
    _hero_positions = _fut_positions
    _hero_max_pos = MAX_POSITIONS
    _hero_badge = f"FUTURES {_fut_leverage}x"
else:
    _hero_equity = _total_equity
    _hero_pnl = _total_pnl
    _hero_trades = _total_trades
    _hero_wins = _total_wins
    _hero_positions = _total_positions
    _hero_max_pos = MAX_POSITIONS * 2
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

tab_market, tab_analysis, tab_signals, tab_trading, tab_backtest, tab_performance = st.tabs([
    f"📈  {t('tab_market')}",
    f"🌐  {t('tab_analysis')}",
    f"🎯  {t('tab_signals')}",
    f"💱  {t('tab_trading')}",
    f"🧪  {t('tab_backtest')}",
    f"⚡  {t('tab_performance')}",
])

# ═══════════════════════════════════════════════════════════════
# SIDEBAR (control panel) — rendered before any tab content
# ═══════════════════════════════════════════════════════════════
render_sidebar(_spot_state, _fut_state)


# ═══════════════════════════════════════════════════════════════
# MARKET TAB
# ═══════════════════════════════════════════════════════════════

with tab_market:
    render_today_summary(_spot_state, _fut_state)

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
        # Single parallel batch instead of 20 sequential fetches (~6s → ~0.5s cold).
        spark_data = fetch_ohlcv_batch(tuple(WATCHED_COINS), "1h", limit=48)
        for idx, coin in enumerate(WATCHED_COINS):
            col = spark_cols[idx % 4]
            spark_df = spark_data.get(coin, pd.DataFrame())
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
# MARKET ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════

with tab_analysis:
    # Manual refresh button (clears all cached data on this tab)
    rcol1, rcol2 = st.columns([10, 1])
    with rcol2:
        if st.button("🔄", help=t("refresh_data"), key="refresh_analysis"):
            fetch_top_movers.clear()
            fetch_funding_rates_batch.clear()
            fetch_fear_greed.clear()
            fetch_btc_dominance.clear()
            st.rerun()

    # ── Market vibe strip ──
    _fng = fetch_fear_greed()
    _dom = fetch_btc_dominance()

    vc1, vc2, vc3, vc4 = st.columns(4)
    if _fng:
        fng_color = "#F6465D" if _fng["value"] < 30 else ("#FCD535" if _fng["value"] < 55 else "#0ECB81")
        vc1.markdown(
            f"""<div style="padding:14px;background:#1E2329;border-radius:8px;">
            <div style="color:#848E9C;font-size:11px;">{t('fng_label')}</div>
            <div style="color:{fng_color};font-size:24px;font-weight:700;">{_fng['value']}</div>
            <div style="color:#EAECEF;font-size:12px;">{_fng['classification']}</div>
            </div>""", unsafe_allow_html=True,
        )
    else:
        vc1.info(f"{t('fng_label')}: n/a")

    if _dom:
        mcap_color = "#0ECB81" if _dom["mcap_change_24h"] >= 0 else "#F6465D"
        vc2.markdown(
            f"""<div style="padding:14px;background:#1E2329;border-radius:8px;">
            <div style="color:#848E9C;font-size:11px;">{t('btc_dom_label')}</div>
            <div style="color:#EAECEF;font-size:24px;font-weight:700;">{_dom['btc_dom']:.1f}%</div>
            <div style="color:#848E9C;font-size:12px;">{t('btc_dom_sub')}</div>
            </div>""", unsafe_allow_html=True,
        )
        vc3.markdown(
            f"""<div style="padding:14px;background:#1E2329;border-radius:8px;">
            <div style="color:#848E9C;font-size:11px;">{t('total_mcap_label')}</div>
            <div style="color:#EAECEF;font-size:24px;font-weight:700;">${_dom['total_mcap_usd']/1e12:.2f}T</div>
            <div style="color:{mcap_color};font-size:12px;">{_dom['mcap_change_24h']:+.2f}% 24h</div>
            </div>""", unsafe_allow_html=True,
        )
    else:
        vc2.info(f"{t('btc_dom_label')}: n/a")
        vc3.info(f"{t('total_mcap_label')}: n/a")

    if _fng:
        if _fng["value"] < 25:
            regime_txt = f"🟢 {t('regime_extreme_fear')}"
            regime_color = "#0ECB81"
        elif _fng["value"] < 45:
            regime_txt = f"🟡 {t('regime_fear')}"
            regime_color = "#FCD535"
        elif _fng["value"] < 65:
            regime_txt = f"⚪ {t('regime_neutral')}"
            regime_color = "#848E9C"
        elif _fng["value"] < 80:
            regime_txt = f"🟠 {t('regime_greed')}"
            regime_color = "#F6465D"
        else:
            regime_txt = f"🔴 {t('regime_extreme_greed')}"
            regime_color = "#F6465D"
        vc4.markdown(
            f"""<div style="padding:14px;background:#1E2329;border-radius:8px;">
            <div style="color:#848E9C;font-size:11px;">{t('regime_hint_label')}</div>
            <div style="color:{regime_color};font-size:13px;font-weight:600;margin-top:4px;">{regime_txt}</div>
            </div>""", unsafe_allow_html=True,
        )

    st.divider()

    # ── Top Gainers / Losers ──
    try:
        _movers = fetch_top_movers(top_n=10)
        _movers_err = None
    except Exception as e:
        _movers = {"gainers": [], "losers": []}
        _movers_err = str(e)

    gc, lc = st.columns(2)
    with gc:
        st.markdown(f"#### 🟢 {t('top_gainers')}")
        if _movers["gainers"]:
            g_df = pd.DataFrame(_movers["gainers"])
            g_df["Price"] = g_df["last"].apply(lambda x: f"${x:,.4f}" if x < 10 else f"${x:,.2f}")
            g_df["24h"] = g_df["pct_24h"].apply(lambda x: f"+{x:.2f}%")
            g_df["Volume"] = g_df["quote_vol"].apply(lambda x: f"${x/1e6:.1f}M")
            st.dataframe(
                g_df[["symbol", "Price", "24h", "Volume"]].rename(columns={"symbol": "Symbol"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info(t("load_error") if _movers_err else t("no_data_available"))
    with lc:
        st.markdown(f"#### 🔴 {t('top_losers')}")
        if _movers["losers"]:
            l_df = pd.DataFrame(_movers["losers"])
            l_df["Price"] = l_df["last"].apply(lambda x: f"${x:,.4f}" if x < 10 else f"${x:,.2f}")
            l_df["24h"] = l_df["pct_24h"].apply(lambda x: f"{x:.2f}%")
            l_df["Volume"] = l_df["quote_vol"].apply(lambda x: f"${x/1e6:.1f}M")
            st.dataframe(
                l_df[["symbol", "Price", "24h", "Volume"]].rename(columns={"symbol": "Symbol"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info(t("load_error") if _movers_err else t("no_data_available"))

    st.divider()

    # ── Watchlist heatmap (24h pct from cached tickers) ──
    st.markdown(f"#### 🌡️ {t('heatmap_title')}")
    if _tickers:
        cells = []
        for sym in WATCHED_COINS:
            tk = _tickers.get(sym)
            if not tk:
                continue
            pct = tk.get("percentage") or 0
            last = tk.get("last") or 0
            cells.append({"sym": sym.replace("/USDT", ""), "pct": pct, "last": last})
        if cells:
            ncols = 5
            rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)]
            for row in rows:
                row_cols = st.columns(ncols)
                for i, cell in enumerate(row):
                    if cell["pct"] >= 5: bg = "#0ECB81"
                    elif cell["pct"] >= 2: bg = "rgba(14,203,129,0.5)"
                    elif cell["pct"] >= 0: bg = "rgba(14,203,129,0.2)"
                    elif cell["pct"] >= -2: bg = "rgba(246,70,93,0.2)"
                    elif cell["pct"] >= -5: bg = "rgba(246,70,93,0.5)"
                    else: bg = "#F6465D"
                    row_cols[i].markdown(
                        f"""<div style="padding:10px;background:{bg};border-radius:6px;text-align:center;">
                        <div style="color:#FFF;font-weight:700;font-size:13px;">{cell['sym']}</div>
                        <div style="color:#FFF;font-size:11px;">{cell['pct']:+.2f}%</div>
                        </div>""", unsafe_allow_html=True,
                    )
        else:
            st.info(t("no_data_available"))
    else:
        st.info(t("no_data_available"))

    st.divider()

    # ── Funding rates for watched coins ──
    st.markdown(f"#### 💰 {t('funding_title')}")
    _fr = fetch_funding_rates_batch(tuple(WATCHED_COINS[:12]))
    if _fr:
        fr_rows = []
        for sym, data in _fr.items():
            rate = data.get("current_rate") if isinstance(data, dict) else None
            if rate is None or not data.get("available", True):
                continue
            fr_rows.append({
                "Symbol": sym,
                "Funding": f"{float(rate)*100:+.4f}%",
                "Avg 24h": f"{float(data.get('avg_rate_24h') or 0)*100:+.4f}%",
                "Trend": data.get("trend", "-"),
                "_rate": float(rate),
            })
        if fr_rows:
            fr_rows.sort(key=lambda r: r["_rate"])
            fr_df = pd.DataFrame(fr_rows).drop(columns=["_rate"])
            fc1, fc2 = st.columns(2)
            fc1.markdown(f"**{t('funding_low')}**")
            fc1.dataframe(fr_df.head(5), use_container_width=True, hide_index=True)
            fc2.markdown(f"**{t('funding_high')}**")
            fc2.dataframe(fr_df.tail(5).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info(t("no_data_available"))
    else:
        st.info(t("no_data_available"))


# ═══════════════════════════════════════════════════════════════
# SIGNALS TAB
# ═══════════════════════════════════════════════════════════════

with tab_signals:
    rcol1, rcol2 = st.columns([10, 1])
    with rcol2:
        if st.button("🔄", help=t("refresh_data"), key="refresh_signals"):
            fetch_live_signals.clear()
            st.session_state.pop("_live_sigs_cache", None)
            st.rerun()

    st.markdown(f"#### 🎯 {t('signals_live_title')} ({t('signals_cache_hint')})")
    st.caption(t("signals_caption"))

    # Defer the heavy SignalGenerator scan: it costs ~10–15s on cold cache
    # and previously ran on every page render even when this tab wasn't
    # open. Stash result in session_state so reruns are instant.
    _live_sigs = st.session_state.get("_live_sigs_cache")
    _show_signals_content = _live_sigs is not None
    if not _show_signals_content:
        scan_col1, scan_col2 = st.columns([3, 1])
        scan_col1.info("ℹ️ Tín hiệu chưa được quét. Bấm nút bên phải — quét mất ~10s lần đầu.")
        if scan_col2.button("🚀 Quét tín hiệu", type="primary", key="scan_signals_btn"):
            with st.spinner(t("signals_scanning")):
                _live_sigs = fetch_live_signals()
            st.session_state["_live_sigs_cache"] = _live_sigs
            st.rerun()

    if not _show_signals_content:
        pass  # Scan button shown above; skip the rest until user opts in
    elif not _live_sigs:
        st.warning(t("signals_no_data"))
    else:
        active = [s for s in _live_sigs if s.get("signal", 0) != 0]
        neutral = [s for s in _live_sigs if s.get("signal", 0) == 0]

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(t("signals_scanned"), len(_live_sigs))
        sc2.metric(t("signals_active"), len(active))
        sc3.metric(t("signals_neutral"), len(neutral))

        st.divider()

        if active:
            st.markdown(f"##### ✅ {t('signals_active_title')}")
            rows = []
            for s in active:
                side = "LONG" if s["signal"] == 1 else "SHORT"
                entry = s.get("close", 0)
                stop_pct = s.get("stop_loss", 0.025)
                if s["signal"] == 1:
                    stop = entry * (1 - stop_pct)
                    tp1 = entry + (entry - stop) * 2
                    tp2 = entry + (entry - stop) * 3
                else:
                    stop = entry * (1 + stop_pct)
                    tp1 = entry - (stop - entry) * 2
                    tp2 = entry - (stop - entry) * 3
                rows.append({
                    "Symbol": s.get("symbol", ""),
                    "Side": side,
                    "Strategy": s.get("strategy", "-"),
                    "Entry": f"${entry:,.4f}" if entry < 10 else f"${entry:,.2f}",
                    "Stop": f"${stop:,.4f}" if stop < 10 else f"${stop:,.2f}",
                    "TP1 (2R)": f"${tp1:,.4f}" if tp1 < 10 else f"${tp1:,.2f}",
                    "TP2 (3R)": f"${tp2:,.4f}" if tp2 < 10 else f"${tp2:,.2f}",
                    "Regime": s.get("regime", "-"),
                    "MTF 1D": s.get("mtf_daily", "-"),
                    "Reason": s.get("reason", "")[:60],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info(t("signals_none_active"))

        if neutral:
            with st.expander(f"{len(neutral)} {t('signals_neutral_expander')}"):
                rows = []
                for s in neutral:
                    rows.append({
                        "Symbol": s.get("symbol", ""),
                        "Strategy": s.get("strategy", "-"),
                        "Regime": s.get("regime", "-"),
                        "MTF 1D": s.get("mtf_daily", "-"),
                        "RSI": f"{s.get('rsi', 0):.1f}" if s.get("rsi") else "-",
                        "ROC10": f"{s.get('roc_10', 0):+.2f}%" if s.get("roc_10") is not None else "-",
                        "ADX": f"{s.get('adx', 0):.1f}" if s.get("adx") else "-",
                        "Reason": s.get("reason", "")[:60],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
        initial_cap = state.get("initial_capital", 500)
        total_trades_count = state["total_trades"]
        total_wins = state["total_wins"]
        open_pos = state["open_positions"]
        open_count = len(open_pos)
        history = state.get("trade_history", [])

        # Equity = capital + MTM of open positions; PnL = equity - initial.
        # Avoids the realized-only accumulator which misses unrealized + entry fees.
        total_equity = capital
        for sym, pos in open_pos.items():
            total_equity += _spot_pos_mtm({**pos, "symbol": pos.get("symbol", sym)})
        total_pnl = total_equity - initial_cap
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
        m3.metric(t("open_positions"), f"{open_count}/{MAX_POSITIONS}")
        m4.metric(t("total_trades"), total_trades_count)
        m5.metric(t("win_rate"), f"{win_rate:.0f}%")
        m6.metric(t("consec_losses"), state.get("consecutive_losses", 0))

        st.divider()

        # --- Open Positions ---
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(f"""
            <div class="section-header" style="margin-top:0;">
                <div><p class="section-title" style="font-size:16px;">{t("open_positions")}</p></div>
            </div>
            """, unsafe_allow_html=True)
            # Reuse the cached _tickers dict (WATCHED_COINS) instead of fresh
            # per-symbol fetch_ticker calls — those were uncached and ran
            # sequentially on every render (~0.5s × open positions).
            _spot_current_prices = {}
            for _sym in open_pos:
                tk = (_tickers or {}).get(_sym)
                if tk and tk.get("last"):
                    _spot_current_prices[_sym] = tk["last"]

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
                        price_html = f'<span class="pos-value">${cur_price:,.4f}</span>'
                        pnl_html = f'<span class="pos-value" style="color:{pnl_color};font-weight:700;">{pnl_pct:+.2f}% (${pnl_usd:+.2f})</span>'
                    else:
                        price_html = '<span class="pos-value" style="color:#848E9C;">--</span>'
                        pnl_html = '<span class="pos-value" style="color:#848E9C;">--</span>'

                    # Build progress bar if we have all the prices we need
                    progress_html = ""
                    distance_html = ""
                    if cur_price and pos.get("tp1_price") and pos.get("tp2_price") and stop_p:
                        progress_html = render_position_progress_bar(
                            entry=entry_p, current=cur_price,
                            stop=stop_p, tp1=pos["tp1_price"], tp2=pos["tp2_price"],
                            side=side,
                        )
                        # Distance to SL/TP in R units
                        risk_per_r = abs(entry_p - stop_p)
                        if risk_per_r > 0:
                            if side == "long":
                                d_sl = (cur_price - stop_p) / risk_per_r
                                d_tp1 = (pos["tp1_price"] - cur_price) / risk_per_r
                                d_tp2 = (pos["tp2_price"] - cur_price) / risk_per_r
                            else:
                                d_sl = (stop_p - cur_price) / risk_per_r
                                d_tp1 = (cur_price - pos["tp1_price"]) / risk_per_r
                                d_tp2 = (cur_price - pos["tp2_price"]) / risk_per_r
                            distance_html = (
                                '<div style="display:flex;justify-content:space-between;'
                                'font-size:11px;color:#848E9C;padding:4px 0;">'
                                f'<span>📍 {abs(d_sl):.2f}R {t("to_stop")}</span>'
                                f'<span>🎯 {abs(d_tp1):.2f}R {t("to_tp1")}</span>'
                                f'<span>🏁 {abs(d_tp2):.2f}R {t("to_tp2")}</span>'
                                '</div>'
                            )

                    # Time held
                    held_html = ""
                    if pos.get("opened_at"):
                        try:
                            opened_dt = datetime.fromisoformat(pos["opened_at"])
                            held_h = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
                            held_html = f'<span style="color:#848E9C;font-size:11px;">⏱️ {t("held_for")} {held_h:.1f}h</span>'
                        except Exception:
                            pass

                    pnl_bg = "14,203,129" if cur_price and pnl_pct >= 0 else "246,70,93"
                    card_html = (
                        '<div class="pos-card">'
                        f'<div class="pos-header">'
                        f'<span class="pos-symbol">{sym}</span>'
                        f'<span class="pos-side {side_cls}">{side_label}</span>'
                        '</div>'
                        f'<div class="pos-detail">'
                        f'<span class="pos-label">{t("current_price")}</span>'
                        f'{price_html}'
                        '</div>'
                        f'<div class="pos-detail" style="background:rgba({pnl_bg},0.05);'
                        f'border-radius:4px;padding:4px 8px;margin:2px 0;">'
                        f'<span class="pos-label">PnL</span>{pnl_html}'
                        '</div>'
                        f'{progress_html}'
                        f'{distance_html}'
                        '<div style="display:flex;justify-content:space-between;font-size:11px;'
                        'color:#848E9C;padding:6px 0 2px 0;border-top:1px solid #2B3139;margin-top:6px;">'
                        f'<span>{t("size")}: {size_str}</span>'
                        f'{held_html}'
                        '</div>'
                        '</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # --- Manual close controls ---
                    pct_key = f"spot_close_pct_{sym}"
                    pct = st.slider(
                        f"{sym} — Close %", 10, 100, 100, 5,
                        key=pct_key, label_visibility="collapsed",
                    )
                    bcol1, bcol2 = st.columns(2)
                    if bcol1.button(t("close_pct").format(pct=pct), key=f"spot_close_btn_{sym}", use_container_width=True):
                        manual_actions.append_close("spot", sym, pct / 100)
                        st.success(t("close_queued").format(pct=pct, sym=sym))
                        st.rerun()
                    if bcol2.button(t("close_all"), key=f"spot_close_all_{sym}",
                                    use_container_width=True, type="primary"):
                        manual_actions.append_close("spot", sym, 1.0)
                        st.success(t("close_queued").format(pct=100, sym=sym))
                        st.rerun()
                    _pending = manual_actions.pending_for("spot", sym)
                    if _pending:
                        st.caption(t("pending_n").format(n=len(_pending)))
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
            f_initial = futures_state.get("initial_capital", 500)
            f_total_trades = futures_state.get("total_trades", 0)
            f_total_wins = futures_state.get("total_wins", 0)
            f_open_pos = futures_state.get("open_positions", {})
            f_open_count = len(f_open_pos)
            f_history = futures_state.get("trade_history", [])
            f_consec_losses = futures_state.get("consecutive_losses", 0)

            # Equity = capital + (margin + uPnL) for each position; PnL = equity - initial.
            f_margin_used = 0
            f_total_equity = f_capital
            f_leverage = 3  # default
            for sym, pos in f_open_pos.items():
                margin = pos.get("margin", pos.get("size_usdt", 0) / pos.get("leverage", 3))
                f_margin_used += margin
                f_total_equity += _fut_pos_mtm({**pos, "symbol": pos.get("symbol", sym)})
                f_leverage = pos.get("leverage", 3)
            f_total_pnl = f_total_equity - f_initial
            f_drawdown = (f_total_equity - f_peak) / f_peak if f_peak > 0 else 0
            f_win_rate = (f_total_wins / f_total_trades * 100) if f_total_trades > 0 else 0

            # --- Metrics ---
            st.markdown(f"#### {t('futures_trading')}")
            fm1, fm2, fm3, fm4, fm5, fm6 = st.columns(6)
            fm1.metric(t("equity"), f"${f_total_equity:.2f}", f"${f_total_pnl:+.2f}")
            fm2.metric(t("leverage"), f"{f_leverage}x")
            fm3.metric(t("margin_used"), f"${f_margin_used:.2f}")
            fm4.metric(t("open_positions"), f"{f_open_count}/{MAX_POSITIONS}")
            fm5.metric(t("win_rate"), f"{f_win_rate:.0f}%")
            fm6.metric(t("drawdown"), f"{f_drawdown:.1%}")

            st.divider()

            # Reuse the cached _tickers dict to avoid uncached per-position
            # fetch_ticker calls on every render.
            _fut_current_prices = {}
            for _sym in f_open_pos:
                tk = (_tickers or {}).get(_sym)
                if tk and tk.get("last"):
                    _fut_current_prices[_sym] = tk["last"]

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

                # --- Progress bar per position (visual SL/entry/TP1/TP2 location) ---
                for fsym, fpos in f_open_pos.items():
                    f_cur = _fut_current_prices.get(fsym)
                    if not (f_cur and fpos.get("tp1_price") and fpos.get("tp2_price") and fpos.get("stop_price")):
                        continue
                    f_entry = fpos.get("entry_price", 0)
                    f_stop = fpos.get("stop_price", 0)
                    risk_per_r = abs(f_entry - f_stop)
                    if fpos["side"] == "long":
                        d_sl = (f_cur - f_stop) / risk_per_r if risk_per_r else 0
                        d_tp1 = (fpos["tp1_price"] - f_cur) / risk_per_r if risk_per_r else 0
                        d_tp2 = (fpos["tp2_price"] - f_cur) / risk_per_r if risk_per_r else 0
                    else:
                        d_sl = (f_stop - f_cur) / risk_per_r if risk_per_r else 0
                        d_tp1 = (f_cur - fpos["tp1_price"]) / risk_per_r if risk_per_r else 0
                        d_tp2 = (f_cur - fpos["tp2_price"]) / risk_per_r if risk_per_r else 0
                    st.markdown(f"**{fsym}** ({fpos['side'].upper()})", help=t("progress_help"))
                    st.markdown(
                        render_position_progress_bar(
                            entry=f_entry, current=f_cur, stop=f_stop,
                            tp1=fpos["tp1_price"], tp2=fpos["tp2_price"],
                            side=fpos["side"],
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div style="display:flex;justify-content:space-between;'
                        'font-size:11px;color:#848E9C;padding:0 0 8px 0;">'
                        f'<span>📍 {abs(d_sl):.2f}R {t("to_stop")}</span>'
                        f'<span>🎯 {abs(d_tp1):.2f}R {t("to_tp1")}</span>'
                        f'<span>🏁 {abs(d_tp2):.2f}R {t("to_tp2")}</span>'
                        '</div>', unsafe_allow_html=True,
                    )

                # --- Manual close controls (one row per futures position) ---
                st.markdown(f"##### {t('manual_close_section')}")
                for fsym in list(f_open_pos.keys()):
                    fcol_lbl, fcol_pct, fcol_b1, fcol_b2 = st.columns([2, 3, 2, 2])
                    fcol_lbl.markdown(f"**{fsym}**")
                    f_pct_key = f"fut_close_pct_{fsym}"
                    f_pct = fcol_pct.slider(
                        f"{fsym} pct", 10, 100, 100, 5,
                        key=f_pct_key, label_visibility="collapsed",
                    )
                    if fcol_b1.button(t("close_pct").format(pct=f_pct), key=f"fut_close_btn_{fsym}", use_container_width=True):
                        manual_actions.append_close("futures", fsym, f_pct / 100)
                        st.success(t("close_queued").format(pct=f_pct, sym=fsym))
                        st.rerun()
                    if fcol_b2.button(t("close_all"), key=f"fut_close_all_{fsym}",
                                      use_container_width=True, type="primary"):
                        manual_actions.append_close("futures", fsym, 1.0)
                        st.success(t("close_queued").format(pct=100, sym=fsym))
                        st.rerun()
                    _fpending = manual_actions.pending_for("futures", fsym)
                    if _fpending:
                        st.caption(t("pending_n").format(n=len(_fpending)))

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

            st.divider()
            render_advanced_breakdowns(history, key_prefix="spot_perf")
            st.divider()
            render_guard_status(state)
        else:
            st.info(t("no_trades"))
            render_guard_status(state)

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

                st.divider()
                render_advanced_breakdowns(f_history, key_prefix="fut_perf")
                st.divider()
                render_guard_status(futures_state)
            else:
                st.info(t("no_trades"))
                render_guard_status(futures_state)
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
