"""
FinRL-X-MT5 Configuration
=========================
Pydantic-based settings for the K-Dense Council MoE trading system.
Reads from environment variables or a .env file.
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


class MT5Settings(BaseSettings):
    """MetaTrader 5 connection & instrument configuration."""

    # Connection
    mt5_login: int = Field(0, description="MT5 account login number")
    mt5_password: str = Field("", description="MT5 account password")
    mt5_server: str = Field("", description="MT5 broker server name")
    mt5_path: str = Field(
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        description="Path to MetaTrader 5 terminal executable",
    )
    mt5_timeout: int = Field(60_000, description="Connection timeout in ms")

    # Instrument universe (exact broker symbol names confirmed from active MT5 terminal)
    symbols: list[str] = Field(
        default=[
            "NAS100.x",
            "WTI.x",
            "XAGUSD.x",
            "US30.x",
            "SPX500.x",
            "GER40.x",
            "JAP225.x",
            "UK100.x",
            "AUS200.x",
            "AAPL.x",
            "NVDA.x",
            "MSFT.x",
            "AMZN.x",
            "META.x",
            "TSLA.x",
            "PLTR.x",
        ],
        description="Active trading symbols matching broker naming convention",
    )

    # Timeframe for Python-side feature engineering (M5 = 5 minutes)
    # MT5 constants: M1=1, M5=5, M15=15, M30=30, H1=16385, H4=16388, D1=16408
    timeframe_minutes: int = Field(5, description="Bar aggregation timeframe in minutes")

    # Position sizing & Prop Firm Safety Presets (Ultra-Safe Profile)
    default_risk_pct: float = Field(0.0025, description="Max risk per trade as % of equity (0.0025 = 0.25% Ultra-Safe)")
    max_portfolio_risk_pct: float = Field(0.02, description="Max total portfolio risk % (2.0% Ultra-Safe cap)")
    max_drawdown_halt_pct: float = Field(0.04, description="Peak-to-trough Drawdown % that halts trading (4.0% hard stop)")
    max_daily_loss_pct: float = Field(0.025, description="Max daily drawdown before circuit breaker trips (2.5% daily stop)")
    min_free_margin_pct: float = Field(0.40, description="Min free margin % before blocking orders (40% free margin guard)")

    # Instrument-specific lot config (verified against broker terminal specifications)
    instrument_config: dict = Field(
        default={
            "NAS100.x":  {"contract_size": 10.0,   "min_lot": 0.01, "lot_step": 0.01, "point": 0.01,  "digits": 2},
            "WTI.x":    {"contract_size": 100.0,  "min_lot": 0.01, "lot_step": 0.01, "point": 0.01,  "digits": 2},
            "XAGUSD.x": {"contract_size": 5000.0, "min_lot": 0.01, "lot_step": 0.01, "point": 0.001, "digits": 3},
            "US30.x":   {"contract_size": 1.0,    "min_lot": 0.01, "lot_step": 0.01, "point": 1.0,   "digits": 0},
            "SPX500.x": {"contract_size": 10.0,   "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.1,   "digits": 1},
            "GER40.x":  {"contract_size": 1.0,    "min_lot": 0.01, "lot_step": 0.01, "point": 0.1,   "digits": 1},
            "JAP225.x": {"contract_size": 100.0,  "min_lot": 0.01, "lot_step": 0.01, "point": 1.0,   "digits": 0},
            "UK100.x":  {"contract_size": 1.0,    "min_lot": 0.01, "lot_step": 0.01, "point": 0.01,  "digits": 2},
            "AUS200.x": {"contract_size": 1.0,    "min_lot": 0.01, "lot_step": 0.01, "point": 1.0,   "digits": 0},
            "AAPL.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "NVDA.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "MSFT.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "AMZN.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "META.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "TSLA.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
            "PLTR.x":   {"contract_size": 1.0,    "min_lot": 0.1,  "lot_step": 0.1,  "point": 0.01,  "digits": 2},
        },
        description="Per-instrument execution parameters from live broker specifications",
    )

    # Magic number for order identification
    magic_number: int = Field(20260908, description="EA magic number for order tagging")
    order_comment: str = Field("FinRL-X-MT5", description="Order comment string")


class DataSettings(BaseSettings):
    """Data source and storage configuration."""

    # Tick history
    tick_lookback_days: int = Field(365, description="Days of tick history to pull for training")
    feature_lookback_bars: int = Field(500, description="M5 bars for feature window")

    # Yahoo Finance (correlation layer)
    # Mapping each MT5 index / commodity to its underlying ETF / equity basket
    yahoo_symbols: dict[str, list[str]] = Field(
        default={
            "NAS100.x": ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
            "SPX500.x": ["SPY", "VOO", "IVV", "XLK", "XLF"],
            "US30.x":   ["DIA", "BA", "GS", "JPM", "UNH", "CAT"],
            "GER40.x":  ["EWG", "SAP"],
            "UK100.x":  ["EWU", "SHEL", "AZN", "HSBC"],
            "JAP225.x": ["EWJ", "DXJ", "TM", "SONY"],
            "AUS200.x": ["EWA", "BHP"],
            "WTI.x":    ["CL=F", "USO", "XLE", "XOM", "CVX"],
            "WTI":      ["CL=F", "USO", "XLE", "XOM", "CVX"],
            "XAGUSD.x": ["SLV", "SI=F", "GLD", "PAAS"],
            "AAPL.x":   ["AAPL", "QQQ", "XLK", "SPY"],
            "NVDA.x":   ["NVDA", "SOXX", "QQQ", "SPY"],
            "MSFT.x":   ["MSFT", "QQQ", "XLK", "SPY"],
            "AMZN.x":   ["AMZN", "XLY", "QQQ", "SPY"],
            "META.x":   ["META", "XLC", "QQQ", "SPY"],
            "TSLA.x":   ["TSLA", "XLY", "QQQ", "SPY"],
            "PLTR.x":   ["PLTR", "QQQ", "XLK", "SPY"],
        },
        description="Yahoo Finance symbols for each MT5 instrument correlation",
    )
    yahoo_interval: str = Field("1d", description="Yahoo Finance data interval")
    yahoo_lookback_days: int = Field(730, description="Days of Yahoo EOD data to pull")

    # FMP API (optional — set key if available)
    fmp_api_key: str = Field("", description="Financial Modeling Prep API key")

    # Storage paths
    data_dir: Path = Field(ROOT / "data", description="Root data directory")
    tick_dir: Path = Field(ROOT / "data" / "ticks", description="Tick cache directory")
    cache_dir: Path = Field(ROOT / "data" / "cache", description="SQLite cache directory")
    db_path: Path = Field(ROOT / "data" / "cache" / "finrl_mt5.db", description="SQLite DB path")


class CouncilSettings(BaseSettings):
    """K-Dense Council MoE configuration."""

    # Expert 1: SAC DRL Agent
    drl_algorithm: Literal["SAC", "PPO", "TD3"] = Field("SAC", description="DRL algorithm")
    drl_total_timesteps: int = Field(1_000_000, description="Total training timesteps")
    drl_n_envs: int = Field(4, description="Parallel training environments")
    drl_lookback_bars: int = Field(50, description="State observation window (bars)")
    drl_learning_rate: float = Field(3e-4)
    drl_buffer_size: int = Field(100_000)
    drl_batch_size: int = Field(256)

    # Expert 2: HMM Regime
    hmm_n_states: int = Field(3, description="Number of market regimes (Bull/Bear/Sideways)")
    hmm_n_iter: int = Field(1000)
    hmm_covariance_type: str = Field("diag")

    # Expert 3: TimesFM
    timesfm_horizon_bars: int = Field(12, description="Forecast horizon in M5 bars (12×5=60 min)")
    timesfm_context_bars: int = Field(512, description="Input context window")
    timesfm_checkpoint: str = Field("google/timesfm-2.5-200m-pytorch")

    # Expert 4: SHAP XGBoost scorer
    xgb_n_estimators: int = Field(300)
    xgb_max_depth: int = Field(6)
    xgb_learning_rate: float = Field(0.05)

    # Expert 5: PyMC Bayesian TP/SL
    pymc_samples: int = Field(1000, description="MCMC posterior samples")
    pymc_tune: int = Field(500, description="MCMC tuning steps")
    pymc_chains: int = Field(2)

    # Council Gate: NSGA-III
    nsga3_pop_size: int = Field(92, description="NSGA-III population size")
    nsga3_n_gen: int = Field(200, description="NSGA-III generations")
    nsga3_n_partitions: int = Field(12, description="Reference direction partitions")
    nsga3_seed: int = Field(42)

    # Model storage
    models_dir: Path = Field(ROOT / "models")
    best_model_dir: Path = Field(ROOT / "models" / "best")
    checkpoint_dir: Path = Field(ROOT / "models" / "checkpoints")
    logs_dir: Path = Field(ROOT / "logs")


class BacktestSettings(BaseSettings):
    """Backtest configuration."""

    # Python-side fast iteration
    bt_start_date: str = Field("2022-01-01", description="Backtest start date (YYYY-MM-DD)")
    bt_end_date: str = Field("2025-01-01", description="Backtest end date (YYYY-MM-DD)")
    bt_initial_cash: float = Field(10_000.0, description="Starting equity in USD")
    bt_commission: float = Field(0.0002, description="Round-trip commission rate")

    # Walk-forward validation
    wf_train_months: int = Field(12, description="Training window in months")
    wf_test_months: int = Field(3, description="OOS test window in months")
    wf_step_months: int = Field(1, description="Walk-forward step size in months")

    # MT5 Strategy Tester — Real Tick target metrics
    target_sharpe: float = Field(2.0)
    target_max_dd: float = Field(0.05, description="5% max drawdown limit")
    target_win_rate: float = Field(0.55)
    target_profit_factor: float = Field(2.0)


class Settings(BaseSettings):
    """Unified top-level settings."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    mt5: MT5Settings = Field(default_factory=MT5Settings)
    data: DataSettings = Field(default_factory=DataSettings)
    council: CouncilSettings = Field(default_factory=CouncilSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)

    # Logging
    log_level: str = Field("INFO")
    log_dir: Path = Field(ROOT / "logs")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


# Singleton — import this everywhere
settings = Settings()
