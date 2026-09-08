"""
SQLite Data Store
==================
Caches MT5 tick features and Yahoo correlation data locally to avoid
repeated API calls. Uses SQLAlchemy for schema management.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

import polars as pl
from loguru import logger

from src.config.settings import settings


class DataStore:
    """SQLite-backed cache for processed feature matrices."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.data.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tick_features (
                    symbol TEXT,
                    timeframe TEXT,
                    date_from TEXT,
                    date_to TEXT,
                    n_bars INTEGER,
                    created_at TEXT,
                    parquet_path TEXT,
                    PRIMARY KEY (symbol, timeframe, date_from, date_to)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS yahoo_features (
                    symbol TEXT,
                    date_from TEXT,
                    date_to TEXT,
                    n_rows INTEGER,
                    created_at TEXT,
                    parquet_path TEXT,
                    PRIMARY KEY (symbol, date_from, date_to)
                )
            """)
            conn.commit()
        logger.debug(f"DataStore initialized at {self.db_path}")

    def save_tick_features(
        self,
        symbol: str,
        features: pl.DataFrame,
        date_from: datetime,
        date_to: datetime,
        timeframe: str = "M5",
    ) -> Path:
        """Save tick feature DataFrame to parquet and register in DB."""
        parquet_dir  = settings.data.cache_dir / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        fname        = f"{symbol}_{timeframe}_{date_from.date()}_{date_to.date()}.parquet"
        parquet_path = parquet_dir / fname

        features.write_parquet(parquet_path)

        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tick_features
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe,
                date_from.isoformat(), date_to.isoformat(),
                len(features), datetime.now().isoformat(),
                str(parquet_path),
            ))
            conn.commit()

        logger.info(f"💾 Cached {len(features):,} bars → {parquet_path.name}")
        return parquet_path

    def load_tick_features(
        self,
        symbol: str,
        date_from: datetime,
        date_to: datetime,
        timeframe: str = "M5",
    ) -> pl.DataFrame | None:
        """Load cached tick features if available."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT parquet_path FROM tick_features
                WHERE symbol=? AND timeframe=?
                  AND date_from<=? AND date_to>=?
            """, (
                symbol, timeframe,
                date_from.isoformat(), date_to.isoformat(),
            )).fetchone()

        if row and Path(row[0]).exists():
            df = pl.read_parquet(row[0])
            logger.info(f"📂 Cache hit: {len(df):,} bars for {symbol} {timeframe}")
            return df
        return None
