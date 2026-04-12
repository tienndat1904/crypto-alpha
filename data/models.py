"""
Database Models - OHLCV Table Definition
=========================================
Defines the schema for storing candlestick data in MySQL.
"""

from sqlalchemy import (
    create_engine, Column, String, Float, BigInteger,
    DateTime, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from config.settings import DATABASE_URL
from loguru import logger

Base = declarative_base()


class OHLCV(Base):
    """
    OHLCV candlestick data table.
    
    Each row = one candle for one symbol at one timeframe.
    Example: BTC/USDT, 4h, 2024-01-01 00:00:00
    """
    __tablename__ = "ohlcv"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)          # e.g. "BTC/USDT"
    timeframe = Column(String(5), nullable=False)         # e.g. "1h", "4h", "1d"
    timestamp = Column(DateTime, nullable=False)          # Candle open time (UTC)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)                # Base asset volume

    # Prevent duplicate candles
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_symbol_tf_ts"),
        Index("idx_symbol_timeframe", "symbol", "timeframe"),
        Index("idx_timestamp", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<OHLCV {self.symbol} {self.timeframe} "
            f"{self.timestamp} C={self.close}>"
        )


# ── Database Engine & Session ──
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified successfully.")


if __name__ == "__main__":
    init_db()
    print("Database initialized!")
