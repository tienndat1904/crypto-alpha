"""
EDA Notebook Generator
======================
Generates 01_eda.ipynb for comprehensive crypto data analysis.
Run: python notebooks/generate_eda.py
"""

import json

def make_cell(cell_type, source, metadata=None):
    """Create a Jupyter cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def md(text):
    return make_cell("markdown", [line + "\n" for line in text.strip().split("\n")])

def code(text):
    return make_cell("code", [line + "\n" for line in text.strip().split("\n")])

cells = []

# ── Title ──
cells.append(md("""# Crypto Alpha — Exploratory Data Analysis
## Phase 1, Tuần 2: EDA & Feature Engineering

**Mục tiêu:**
- Hiểu đặc điểm dữ liệu từng coin (phân phối, volatility, correlation)
- Tính technical indicators & engineered features
- Xác định volatility regimes và market microstructure
- Tìm insight sơ bộ cho alpha research (Phase 2)"""))

# ── Setup ──
cells.append(md("## 1. Setup & Load Data"))

cells.append(code("""import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

from config.settings import DATABASE_URL, COIN_UNIVERSE, TIMEFRAMES
from data.models import engine
from utils.indicators import add_all_indicators

plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', 50)

print("Setup complete!")"""))

# ── Load Data ──
cells.append(md("### 1.1. Load dữ liệu từ MySQL"))

cells.append(code("""def load_data(symbol, timeframe):
    \"\"\"Load OHLCV data for a symbol/timeframe from MySQL.\"\"\"
    query = text(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM ohlcv "
        "WHERE symbol = :symbol AND timeframe = :timeframe "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol, "timeframe": timeframe})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df

def load_all_daily():
    \"\"\"Load daily data for all coins into a dict.\"\"\"
    data = {}
    for symbol in COIN_UNIVERSE:
        try:
            df = load_data(symbol, "1d")
            if len(df) > 0:
                data[symbol] = df
                print(f"  {symbol}: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
    return data

print(f"Loading daily data for {len(COIN_UNIVERSE)} coins...")
daily_data = load_all_daily()
print(f"\\nLoaded {len(daily_data)} coins successfully.")"""))

# ── Section 2: Overview ──
cells.append(md("## 2. Data Overview"))

cells.append(md("### 2.1. Thống kê tổng quát"))

cells.append(code("""# Build summary table
summary_rows = []
for symbol, df in daily_data.items():
    summary_rows.append({
        "Symbol": symbol.replace("/USDT", ""),
        "Days": len(df),
        "First": df.index[0].strftime("%Y-%m-%d"),
        "Last": df.index[-1].strftime("%Y-%m-%d"),
        "Min Price": f"${df['close'].min():,.2f}",
        "Max Price": f"${df['close'].max():,.2f}",
        "Current": f"${df['close'].iloc[-1]:,.2f}",
        "Avg Daily Vol ($M)": f"{(df['volume'] * df['close']).mean() / 1e6:,.1f}",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df"""))

# ── Section 3: Price Charts ──
cells.append(md("## 3. Price Action Analysis"))

cells.append(md("### 3.1. Normalized price performance (all coins)"))

cells.append(code("""# Normalize all prices to start at 100 for comparison
fig = go.Figure()

for symbol, df in daily_data.items():
    normalized = (df["close"] / df["close"].iloc[0]) * 100
    name = symbol.replace("/USDT", "")
    fig.add_trace(go.Scatter(
        x=df.index, y=normalized,
        mode="lines", name=name,
        line=dict(width=1.5),
    ))

fig.update_layout(
    title="Normalized Price Performance (Base = 100)",
    xaxis_title="Date",
    yaxis_title="Normalized Price",
    height=500,
    hovermode="x unified",
    legend=dict(font=dict(size=10)),
)
fig.show()"""))

cells.append(md("### 3.2. BTC vs Top Altcoins"))

cells.append(code("""top_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

fig = make_subplots(
    rows=len(top_coins), cols=1,
    shared_xaxes=True,
    subplot_titles=[s.replace("/USDT", "") for s in top_coins],
    vertical_spacing=0.03,
)

for i, symbol in enumerate(top_coins):
    if symbol not in daily_data:
        continue
    df = daily_data[symbol]
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name=symbol.replace("/USDT", ""),
            showlegend=False,
        ),
        row=i+1, col=1,
    )

fig.update_layout(height=1200, title="Price Action — Top 5 Coins")
fig.update_xaxes(rangeslider_visible=False)
fig.show()"""))

# ── Section 4: Returns Analysis ──
cells.append(md("## 4. Returns Distribution Analysis"))

cells.append(md("### 4.1. Phân phối daily returns"))

cells.append(code("""fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, (symbol, df) in enumerate(daily_data.items()):
    if i >= 20:
        break
    returns = df["close"].pct_change().dropna()
    name = symbol.replace("/USDT", "")
    
    axes[i].hist(returns, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[i].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[i].set_title(f"{name}", fontsize=11)
    axes[i].set_xlabel("")
    
    # Add stats
    stats_text = f"μ={returns.mean():.4f}\\nσ={returns.std():.4f}\\nskew={returns.skew():.2f}"
    axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                 fontsize=8, verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle("Daily Returns Distribution", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("### 4.2. Returns statistics table"))

cells.append(code("""returns_stats = []
for symbol, df in daily_data.items():
    r = df["close"].pct_change().dropna()
    returns_stats.append({
        "Symbol": symbol.replace("/USDT", ""),
        "Mean Daily (%)": f"{r.mean()*100:.3f}",
        "Std Daily (%)": f"{r.std()*100:.2f}",
        "Sharpe (ann.)": f"{(r.mean() / r.std()) * np.sqrt(365):.2f}",
        "Skewness": f"{r.skew():.2f}",
        "Kurtosis": f"{r.kurt():.2f}",
        "Max Drawdown (%)": f"{((df['close'] / df['close'].cummax()) - 1).min()*100:.1f}",
        "Best Day (%)": f"{r.max()*100:.1f}",
        "Worst Day (%)": f"{r.min()*100:.1f}",
    })

returns_df = pd.DataFrame(returns_stats)
returns_df.sort_values("Sharpe (ann.)", ascending=False)"""))

# ── Section 5: Correlation ──
cells.append(md("## 5. Correlation Analysis"))

cells.append(md("### 5.1. Correlation matrix (daily returns)"))

cells.append(code("""# Build returns matrix
returns_matrix = pd.DataFrame()
for symbol, df in daily_data.items():
    name = symbol.replace("/USDT", "")
    returns_matrix[name] = df["close"].pct_change()

corr = returns_matrix.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Correlation Matrix — Daily Returns",
    height=700, width=750,
)
fig.show()"""))

cells.append(md("""### 5.2. Rolling correlation với BTC
BTC thường dẫn dắt thị trường crypto. Rolling correlation cho thấy mức độ phụ thuộc thay đổi theo thời gian."""))

cells.append(code("""if "BTC/USDT" in daily_data:
    btc_returns = daily_data["BTC/USDT"]["close"].pct_change()
    
    fig = go.Figure()
    alts = ["ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "LINK/USDT"]
    
    for symbol in alts:
        if symbol not in daily_data:
            continue
        alt_returns = daily_data[symbol]["close"].pct_change()
        # Align indices
        aligned = pd.DataFrame({"btc": btc_returns, "alt": alt_returns}).dropna()
        rolling_corr = aligned["btc"].rolling(30).corr(aligned["alt"])
        
        fig.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr,
            mode="lines", name=symbol.replace("/USDT", ""),
            line=dict(width=1.5),
        ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="30-Day Rolling Correlation with BTC",
        yaxis_title="Correlation",
        height=400,
        hovermode="x unified",
    )
    fig.show()"""))

# ── Section 6: Volatility ──
cells.append(md("## 6. Volatility Analysis"))

cells.append(md("### 6.1. Rolling volatility comparison"))

cells.append(code("""fig = go.Figure()

for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "INJ/USDT"]:
    if symbol not in daily_data:
        continue
    df = daily_data[symbol]
    vol = df["close"].pct_change().rolling(30).std() * np.sqrt(365) * 100
    fig.add_trace(go.Scatter(
        x=df.index, y=vol,
        mode="lines", name=symbol.replace("/USDT", ""),
        line=dict(width=1.5),
    ))

fig.update_layout(
    title="30-Day Rolling Annualized Volatility (%)",
    yaxis_title="Volatility (%)",
    height=400,
    hovermode="x unified",
)
fig.show()"""))

cells.append(md("### 6.2. Volatility regime detection"))

cells.append(code("""# Simple regime: Low / Medium / High volatility based on percentiles
if "BTC/USDT" in daily_data:
    btc = daily_data["BTC/USDT"].copy()
    btc["vol_30d"] = btc["close"].pct_change().rolling(30).std() * np.sqrt(365)
    btc = btc.dropna()
    
    q33 = btc["vol_30d"].quantile(0.33)
    q66 = btc["vol_30d"].quantile(0.66)
    
    btc["regime"] = pd.cut(
        btc["vol_30d"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low", "Medium", "High"]
    )
    
    colors = {"Low": "green", "Medium": "orange", "High": "red"}
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(
        x=btc.index, y=btc["close"],
        mode="lines", name="BTC Price", line=dict(color="steelblue", width=1.5)
    ), row=1, col=1)
    
    for regime in ["Low", "Medium", "High"]:
        mask = btc["regime"] == regime
        fig.add_trace(go.Scatter(
            x=btc.index[mask], y=btc["vol_30d"][mask],
            mode="markers", name=f"{regime} Vol",
            marker=dict(color=colors[regime], size=3),
        ), row=2, col=1)
    
    fig.update_layout(
        title="BTC: Price & Volatility Regime",
        height=600,
    )
    fig.update_yaxes(title_text="Price ($)", row=1)
    fig.update_yaxes(title_text="Ann. Volatility", row=2)
    fig.show()
    
    print(f"\\nVolatility Regime Breakdown:")
    print(btc["regime"].value_counts().to_string())"""))

# ── Section 7: Volume ──
cells.append(md("## 7. Volume Analysis"))

cells.append(code("""# Volume profile for top coins
fig = go.Figure()

for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
    if symbol not in daily_data:
        continue
    df = daily_data[symbol]
    vol_usd = (df["volume"] * df["close"]) / 1e6  # in millions
    vol_sma = vol_usd.rolling(20).mean()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_sma,
        mode="lines", name=symbol.replace("/USDT", ""),
        line=dict(width=1.5),
    ))

fig.update_layout(
    title="20-Day Average Daily Volume ($ Millions)",
    yaxis_title="Volume ($M)",
    height=400,
    hovermode="x unified",
)
fig.show()"""))

# ── Section 8: Feature Engineering ──
cells.append(md("## 8. Feature Engineering — Technical Indicators"))

cells.append(md("### 8.1. Compute tất cả indicators cho BTC (4H timeframe)"))

cells.append(code("""# Load 4H data for BTC - this is our primary trading timeframe
btc_4h = load_data("BTC/USDT", "4h")
print(f"BTC/USDT 4H: {len(btc_4h)} candles")

# Add all indicators
btc_4h = add_all_indicators(btc_4h)
print(f"\\nTotal columns: {len(btc_4h.columns)}")
print(f"\\nFeature list:")
for col in btc_4h.columns:
    print(f"  - {col}")"""))

cells.append(md("### 8.2. Indicators visualization — BTC 4H"))

cells.append(code("""# Last 500 candles for better visibility
df_plot = btc_4h.tail(500)

fig = make_subplots(
    rows=5, cols=1, shared_xaxes=True,
    row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],
    vertical_spacing=0.02,
    subplot_titles=["Price + Bollinger Bands", "RSI", "MACD", "ADX", "Volume"],
)

# Price + BB
fig.add_trace(go.Candlestick(
    x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
    low=df_plot["low"], close=df_plot["close"], name="Price", showlegend=False
), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_upper"],
    line=dict(color="gray", width=0.5, dash="dash"), name="BB Upper", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_lower"],
    line=dict(color="gray", width=0.5, dash="dash"), name="BB Lower",
    fill="tonexty", fillcolor="rgba(128,128,128,0.1)", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["ema_21"],
    line=dict(color="orange", width=1), name="EMA 21"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["rsi"],
    line=dict(color="purple", width=1), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd"],
    line=dict(color="blue", width=1), name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["macd_signal"],
    line=dict(color="red", width=1), name="Signal"), row=3, col=1)
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["macd_hist"],
    name="Histogram", marker_color="gray", opacity=0.5), row=3, col=1)

# ADX
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["adx"],
    line=dict(color="brown", width=1), name="ADX"), row=4, col=1)
fig.add_hline(y=25, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)

# Volume
colors = ["green" if c >= o else "red" for c, o in zip(df_plot["close"], df_plot["open"])]
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["volume"],
    marker_color=colors, opacity=0.6, name="Volume", showlegend=False), row=5, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["volume_sma_20"],
    line=dict(color="blue", width=1), name="Vol SMA 20"), row=5, col=1)

fig.update_layout(height=1200, title="BTC/USDT 4H — Technical Analysis Dashboard")
fig.update_xaxes(rangeslider_visible=False)
fig.show()"""))

# ── Section 9: Feature Correlation ──
cells.append(md("## 9. Feature Correlation with Future Returns"))

cells.append(md("""### 9.1. Tìm features có predictive power
Đo correlation giữa mỗi feature và returns tương lai (next 1, 3, 5 candles).
Đây là bước sơ bộ cho alpha research."""))

cells.append(code("""# Compute forward returns
btc_4h["fwd_return_1"] = btc_4h["close"].pct_change(1).shift(-1)
btc_4h["fwd_return_3"] = btc_4h["close"].pct_change(3).shift(-3)
btc_4h["fwd_return_5"] = btc_4h["close"].pct_change(5).shift(-5)

# Select numeric features (exclude OHLCV and forward returns)
exclude = ["open", "high", "low", "close", "volume",
           "fwd_return_1", "fwd_return_3", "fwd_return_5"]
features = [c for c in btc_4h.select_dtypes(include=[np.number]).columns
            if c not in exclude]

# Compute correlations
corr_results = []
for feat in features:
    for fwd in ["fwd_return_1", "fwd_return_3", "fwd_return_5"]:
        c = btc_4h[[feat, fwd]].dropna().corr().iloc[0, 1]
        corr_results.append({
            "Feature": feat,
            "Forward Return": fwd.replace("fwd_return_", "") + " candles",
            "Correlation": c,
            "Abs Corr": abs(c),
        })

corr_df = pd.DataFrame(corr_results)

# Top features by absolute correlation with 1-candle forward return
top_features = (
    corr_df[corr_df["Forward Return"] == "1 candles"]
    .sort_values("Abs Corr", ascending=False)
    .head(20)
)

print("Top 20 features correlated with next-candle return (BTC 4H):")
print(top_features[["Feature", "Correlation", "Abs Corr"]].to_string(index=False))"""))

cells.append(code("""# Heatmap of top features vs forward returns
pivot = corr_df.pivot(index="Feature", columns="Forward Return", values="Correlation")
top_feat_names = top_features["Feature"].tolist()
pivot_top = pivot.loc[top_feat_names]

fig = px.imshow(
    pivot_top,
    text_auto=".3f",
    color_continuous_scale="RdBu_r",
    zmin=-0.15, zmax=0.15,
    title="Feature-Forward Return Correlation (BTC 4H)",
    height=600,
)
fig.show()"""))

# ── Section 10: Key Insights ──
cells.append(md("""## 10. Key Insights & Next Steps

### Ghi chú phân tích:
Sau khi chạy notebook, hãy ghi lại:
1. **Coins nào có Sharpe cao nhất?** → Ưu tiên trade
2. **Coins nào ít correlated với BTC?** → Diversification
3. **Volatility regime hiện tại?** → Điều chỉnh strategy
4. **Features nào có predictive power?** → Input cho alpha (Phase 2)
5. **Volume trend** → Thanh khoản đủ để trade?

### Tiếp theo (Phase 2):
- Dùng top features làm input cho alpha ideation
- Backtest các alpha strategies dựa trên insights từ EDA
- Walk-forward validation để tránh overfitting"""))

# ── Build Notebook ──
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = "/home/claude/crypto-alpha/notebooks/01_eda.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook created: {output_path}")
