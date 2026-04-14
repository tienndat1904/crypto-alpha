"""
Crypto Alpha Trading System - Configuration
============================================
Central config for data pipeline, trading params, and risk management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for d in [DATA_DIR, LOG_DIR, NOTEBOOK_DIR]:
    d.mkdir(exist_ok=True)

# ═══════════════════════════════════════════
# EXCHANGE CONFIG
# ═══════════════════════════════════════════
EXCHANGE = "binance"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# ═══════════════════════════════════════════
# DATABASE CONFIG (MySQL)
# ═══════════════════════════════════════════
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "crypto_alpha"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    f"?charset=utf8mb4"
)

# ═══════════════════════════════════════════
# UNIVERSE - Coins to track
# ═══════════════════════════════════════════
# Top 20 by market cap + liquidity, paired with USDT
COIN_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOGE/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "POL/USDT",
    "UNI/USDT",
    "ATOM/USDT",
    "LTC/USDT",
    "FIL/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "NEAR/USDT",
    "INJ/USDT",
]

# ═══════════════════════════════════════════
# DATA PIPELINE CONFIG
# ═══════════════════════════════════════════
TIMEFRAMES = ["1h", "4h", "1d"]           # Timeframes to fetch
HISTORY_DAYS = 730                         # 2 years of historical data
FETCH_BATCH_SIZE = 1000                    # Max candles per API request (Binance limit)
UPDATE_INTERVAL_HOURS = 4                  # Auto-update frequency

# ═══════════════════════════════════════════
# RISK MANAGEMENT (Phase 5-6, defined early)
# ═══════════════════════════════════════════
INITIAL_CAPITAL = 500                      # USD
MAX_RISK_PER_TRADE = 0.02                  # 2% of capital
MAX_POSITIONS = 3                          # Simultaneous positions
MAX_DRAWDOWN = 0.15                        # 15% -> kill switch
MAX_CORRELATED_POSITIONS = 2              # Max same-sector positions
STOP_LOSS_ATR_MULTIPLIER = 1.5            # SL = 1.5x ATR
CORRELATION_THRESHOLD = 0.75              # Block new position if corr > 0.75 with existing
CORRELATION_LOOKBACK_DAYS = 30            # Days to compute rolling correlation

# Trading costs (Binance spot, no BNB discount)
TRADING_FEE = 0.001                        # 0.1% maker/taker
SLIPPAGE_PCT = 0.0005                      # 0.05% estimated slippage per trade

# ═══════════════════════════════════════════
# FUTURES CONFIG
# ═══════════════════════════════════════════
TRADING_MODE = os.getenv("TRADING_MODE", "spot")  # "spot" or "futures"
FUTURES_LEVERAGE = int(os.getenv("FUTURES_LEVERAGE", "3"))  # Max 3x for safety
FUTURES_FEE = 0.0004                       # 0.04% maker on futures
FUTURES_MARGIN_TYPE = "isolated"           # isolated or cross

# ═══════════════════════════════════════════
# ALPHA RESEARCH CONFIG (Phase 2)
# ═══════════════════════════════════════════
MIN_SHARPE_RATIO = 0.5                     # Minimum to keep an alpha
MAX_ALPHA_DRAWDOWN = 0.25                  # Max drawdown per alpha
WALK_FORWARD_TRAIN_RATIO = 0.7            # 70% train, 30% test

# ═══════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FILE = LOG_DIR / "crypto_alpha.log"