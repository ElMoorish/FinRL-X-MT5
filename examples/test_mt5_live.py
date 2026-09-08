"""
Test MT5 Live Data Connection & Feature Pipeline
"""

from __future__ import annotations

import MetaTrader5 as mt5
from loguru import logger

from src.config.settings import settings
from src.data.mt5_tick_fetcher import MT5TickFetcher
from src.data.tick_feature_engineer import TickFeatureEngineer
from src.data.yahoo_fetcher import YahooFetcher
from src.data.correlation_fuser import CorrelationFuser


def main():
    logger.info("Connecting to live MT5 terminal...")
    with MT5TickFetcher() as fetcher:
        account = mt5.account_info()
        logger.info(f"Account: {account.login} | Server: {account.server} | Equity: ${account.equity:,.2f}")

        test_symbols = ["NAS100.x", "WTI.x", "XAGUSD.x", "US30.x", "GER40.x", "SPX500.x"]

        for sym in test_symbols:
            info = fetcher.get_symbol_info(sym)
            if info:
                logger.info(
                    f"Symbol: {sym:<10} | Contract: {info['contract_size']:>6} | "
                    f"Digits: {info['digits']} | Point: {info['point']} | Spread: {info['spread']} pts"
                )
            else:
                logger.warning(f"Could not fetch symbol info for {sym}")

        logger.info("Fetching M5 bars for NAS100.x...")
        bars = fetcher.get_ohlcv("NAS100.x", timeframe=mt5.TIMEFRAME_M5, n_bars=100)
        logger.info(f"Retrieved {len(bars)} M5 bars for NAS100.x")
        logger.info(f"Columns: {bars.columns}")
        logger.info(f"Latest bar time: {bars['time'][-1]} | Close: {bars['close'][-1]}")

        logger.info("Running TickFeatureEngineer...")
        engineer = TickFeatureEngineer()
        feat_bars = engineer.compute_features(bars)
        logger.info(f"Feature engineering complete! Shape: {feat_bars.shape}")

        logger.info("Fetching Yahoo correlation basket for NAS100.x...")
        yahoo = YahooFetcher()
        yahoo_features = yahoo.get_correlation_features("NAS100.x", days=30)
        logger.info(f"Yahoo features retrieved! Shape: {yahoo_features.shape}")

        logger.info("Testing CorrelationFuser...")
        fuser = CorrelationFuser()
        fused = fuser.fuse(feat_bars, yahoo_features)
        logger.info(f"Data fusion complete! Final fused shape: {fused.shape}")
        logger.info("🎉 MT5 LIVE DATA PIPELINE VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
