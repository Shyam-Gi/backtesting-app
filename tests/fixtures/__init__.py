"""Test fixtures module for backtesting system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from backtesting_system.data_loader import DataLoader, ParquetDataStore
from backtesting_system.strategy import BaseStrategy
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    n_days = len(dates)
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible tests
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df_data = {
        "timestamp": dates,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, n_days)
    }
    
    return pl.from_pandas(pd.DataFrame(df_data))


@pytest.fixture
def small_ohlcv_data():
    """Create small sample OHLCV DataFrame for quick tests."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
    
    df = pl.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000],
    })
    
    return df


@pytest.fixture
def data_store(temp_dir):
    """Create ParquetDataStore instance with temporary directory."""
    return ParquetDataStore(data_dir=temp_dir)


@pytest.fixture
def data_loader(data_store):
    """Create DataLoader instance with test data store."""
    return DataLoader(store=data_store)


@pytest.fixture
def sample_strategies():
    """Create sample strategy instances for testing."""
    return {
        "sma": SMAStrategy(fast_period=10, slow_period=50),
        "momentum": MomentumStrategy(lookback_period=20, threshold=0.02),
        "mean_reversion": MeanReversionStrategy(ma_period=20, threshold=0.05)
    }


@pytest.fixture
def backtest_config():
    """Create sample backtest configuration."""
    return {
        "initial_cash": 100000,
        "commission_bps": 5,
        "slippage_pct": 0.001,
        "position_size_pct": 1.0
    }


@pytest.fixture
def trade_signals():
    """Create sample trading signals for testing."""
    return pl.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "signal": [0, 1, 0, 0, -1, 0, 1, 0, 0, -1]  # 1=BUY, -1=SELL, 0=HOLD
    })


@pytest.fixture
def performance_benchmark_data():
    """Create large dataset for performance benchmarking."""
    dates = pd.date_range(start="2015-01-01", end="2024-12-31", freq="D")
    n_days = len(dates)
    
    # Generate 10 years of realistic price data
    np.random.seed(123)
    base_price = 50.0
    returns = np.random.normal(0.0003, 0.015, n_days)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df_data = {
        "timestamp": dates,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        "close": prices,
        "volume": np.random.randint(500000, 2000000, n_days)
    }
    
    return pl.from_pandas(pd.DataFrame(df_data))


@pytest.fixture
def corrupted_ohlcv_data():
    """Create corrupted OHLCV data for error testing."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    
    return pl.DataFrame({
        "timestamp": dates,
        "open": [100.0, None, 102.0, 103.0, 104.0],  # None value
        "high": [101.0, 102.0, 101.0, 104.0, 105.0],  # High < Open violation
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000000, 1100000, -1200000, 1300000, 1400000]  # Negative volume
    })


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, signals=None):
        super().__init__()
        self.signals = signals or [0] * 10
        self.signal_index = 0
    
    def generate_signals(self, df):
        """Generate mock signals."""
        signals = self.signals[:len(df)] + [0] * max(0, len(df) - len(self.signals))
        return df.with_columns(
            pl.Series("signal", signals)
        )


@pytest.fixture
def mock_strategy():
    """Create mock strategy instance."""
    return MockStrategy([0, 1, 0, 0, -1, 0, 1, 0, 0, -1])