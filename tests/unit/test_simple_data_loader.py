"""Simple unit tests for data loader without complex imports."""

import pytest
import tempfile
import shutil
from pathlib import Path
import polars as pl

from backtesting_system.data_loader import DataLoader, ParquetDataStore


class TestSimpleDataLoader:
    """Simple test for data loader functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = ['2024-01-01', '2024-01-02', '2024-01-03']
        return pl.DataFrame({
            "timestamp": dates,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000000, 1100000, 1200000],
        })

    def test_basic_functionality(self):
        """Test basic functionality without complex imports."""
        # Test that we can import the classes
        assert DataLoader is not None
        assert ParquetDataStore is not None
        
        # Test that classes can be instantiated
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParquetDataStore(data_dir=temp_dir)
            loader = DataLoader(store=store)
            
            assert store is not None
            assert loader is not None

    def test_parquet_store_basic(self, temp_dir, sample_df):
        """Test basic ParquetDataStore functionality."""
        store = ParquetDataStore(data_dir=temp_dir)
        symbol = "TEST"
        
        # Test save
        store.save(symbol, sample_df)
        
        # Test load
        loaded_df = store.load(symbol)
        
        # Basic validation
        assert loaded_df.height == sample_df.height
        assert set(loaded_df.columns) == set(sample_df.columns)

    def test_imports_work(self):
        """Test that basic imports work correctly."""
        try:
            from backtesting_system.data_loader import DataLoader, ParquetDataStore
            from backtesting_system.data_store import DataStore
            success = True
        except ImportError as e:
            success = False
            print(f"Import error: {e}")
        
        assert success, "Basic imports should work"