# Crypto Alpha Trading System

Automated cryptocurrency trading system with technical analysis strategies, backtesting engine, and real-time paper trading for both **Spot** (long-only) and **Futures** (long+short, leveraged).

## Features

- **Paper Trading** - Automated spot & futures paper trading with real Binance data
- **Backtesting Engine** - Walk-forward validation, separate spot/futures pipelines
- **Trailing Stop Loss** - 3-phase system: fixed stop -> breakeven -> trailing
- **Risk Management** - Position sizing, correlation filter, regime detection, drawdown protection
- **Watchdog** - Auto-restart bots on crash with Telegram alerts
- **Dashboard** - Streamlit web dashboard with real-time PnL, candlestick charts, favorites
- **Telegram Bot** - Trade alerts, daily reports, interactive commands (/status, /positions, /pnl)
- **On-chain Signals** - Fear & Greed index filter for entry confirmation

## Project Structure

```
crypto-alpha/
├── backtest/              # Backtesting engine (leverage, walk-forward)
├── config/
│   └── settings.py        # Central configuration
├── data/
│   ├── fetcher.py         # Binance OHLCV data fetcher
│   └── models.py          # Database models (MySQL)
├── strategies/
│   ├── technical_alphas.py    # Strategy definitions (Momentum Reversal, etc.)
│   ├── onchain_alphas.py      # On-chain signal filters
│   ├── run_backtest.py        # Spot backtest runner (long-only)
│   └── run_backtest_futures.py # Futures backtest runner (long+short)
├── trading/
│   ├── paper_trader.py    # Spot paper trading bot
│   ├── futures_trader.py  # Futures paper trading bot
│   ├── risk_manager.py    # Risk management & trailing stops
│   ├── signal_generator.py # Signal generation with MTF filter
│   └── price_monitor.py   # WebSocket price monitoring
├── utils/
│   ├── telegram.py        # Telegram alert system
│   ├── indicators.py      # Technical indicators
│   └── universe_scanner.py # Coin universe scanner
├── models/                # ML model artifacts
├── tests/                 # Unit tests
├── dashboard.py           # Streamlit dashboard
├── bot_watchdog.py            # Bot watchdog (auto-restart)
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Binance API
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Database (MySQL)
DATABASE_URL=mysql+pymysql://user:pass@localhost/crypto_alpha

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Run Paper Trading

```bash
# Run via watchdog (recommended - auto-restart on crash)
python bot_watchdog.py

# Or run individually
python -m trading.paper_trader --run --interval 0.25
python -m trading.futures_trader --run --interval 0.25
```

### 4. Run Dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

### 5. Run Backtest

```bash
# Spot backtest (long-only)
python -m strategies.run_backtest

# Futures backtest (long+short, leveraged)
python -m strategies.run_backtest_futures

# Single symbol
python -m strategies.run_backtest --symbol BTC/USDT --strategy momentum_reversal
```

## Trading Strategies

| Strategy | Spot | Futures | Description |
|----------|------|---------|-------------|
| Momentum Reversal | Yes | Yes | ROC oversold + support level + volume confirmation |

*Volatility Breakout and Trend Following disabled after backtest showed negative Sharpe.*

## Risk Management

- **Position Sizing**: Risk-based (max 2% capital per trade)
- **Stop Loss**: ATR-adjusted, regime-aware
- **Trailing Stop** (3 phases):
  - Profit < 1.5%: Fixed stop loss
  - Profit >= 1.5%: Stop moves to breakeven (entry price)
  - Profit >= 2.5%: Stop trails behind best price (1.5% or 2.5x ATR)
- **Take Profit**: TP1 at 2R (partial 50%), TP2 at 3R (full close)
- **Correlation Filter**: Max 0.7 correlation between open positions
- **Kill Switch**: Halt trading at 15% drawdown

## Coin Universe

BTC, ETH, BNB, XRP, SOL, ADA, AVAX, DOGE, DOT, LINK, UNI, ATOM, LTC, FIL, APT, ARB, OP, NEAR, INJ, POL (all paired with USDT)

## Telegram Commands

| Command | Description |
|---------|-------------|
| /status | Account overview |
| /balance | Detailed balance |
| /positions | Open positions with unrealized PnL |
| /history | Last 10 trades |
| /pnl | PnL analysis |
| /report | Daily report |

## Dashboard

Access at `http://localhost:8501` after starting Streamlit.

- **Market Overview**: Price changes, volume, favorite coins with star icons
- **Trading**: Open positions with real-time PnL, candlestick charts with entry/SL/TP lines
- **Backtest**: Strategy comparison with spot/futures toggle, equity curves
- **Settings**: Language (Vietnamese/English), timezone (UTC+7)

## License

Private project. All rights reserved.
