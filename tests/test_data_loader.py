"""Unit tests for data loader and data store."""

import pytest
import tempfile
import shutil
from pathlib import Path
import polars as pl
from backtesting_system.data_loader import DataLoader, ParquetDataStore
from backtesting_system.data_store import DataStore
import time


class TestParquetDataStore:
    """Test Parquet data store backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create ParquetDataStore instance."""
        return ParquetDataStore(data_dir=temp_dir)

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        import pandas as pd
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        df = pl.from_pandas(
            pd.DataFrame({
                "timestamp": dates,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000],
            })
        )
        return df

    def test_save_and_load(self, store, sample_df):
        """Test saving and loading data."""
        store.save("AAPL", sample_df)
        loaded_df = store.load("AAPL")
        assert loaded_df.shape == sample_df.shape
        assert loaded_df.columns == sample_df.columns

    def test_load_with_date_filter(self, store, sample_df):
        """Test loading with date range filter."""
        store.save("AAPL", sample_df)
        loaded_df = store.load("AAPL", start_date="2024-01-05", end_date="2024-01-08")
        assert len(loaded_df) == 4

    def test_exists(self, store, sample_df):
        """Test existence check."""
        assert not store.exists("AAPL")
        store.save("AAPL", sample_df)
        assert store.exists("AAPL")

    def test_delete(self, store, sample_df):
        """Test deletion."""
        store.save("AAPL", sample_df)
        assert store.exists("AAPL")
        store.delete("AAPL")
        assert not store.exists("AAPL")

    def test_overwrite_protection(self, store, sample_df):
        """Test overwrite protection."""
        store.save("AAPL", sample_df)
        with pytest.raises(FileExistsError):
            store.save("AAPL", sample_df, overwrite=False)

    def test_overwrite_allowed(self, store, sample_df):
        """Test overwrite when allowed."""
        store.save("AAPL", sample_df)
        store.save("AAPL", sample_df, overwrite=True)
        assert store.exists("AAPL")

    def test_get_available_symbols(self, store, sample_df):
        """Test listing available symbols."""
        assert len(store.get_available_symbols()) == 0
        store.save("AAPL", sample_df)
        store.save("MSFT", sample_df)
        symbols = store.get_available_symbols()
        assert set(symbols) == {"AAPL", "MSFT"}

    def test_get_date_range(self, store, sample_df):
        """Test date range retrieval."""
        store.save("AAPL", sample_df)
        start, end = store.get_date_range("AAPL")
        assert "2024-01-01" in start
        assert "2024-01-10" in end


class TestDataLoader:
    """Test DataLoader with yfinance integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_dir):
        """Create DataLoader instance."""
        return DataLoader(data_dir=temp_dir)

    def test_download_real_data(self, loader):
        """Test downloading real data from yfinance.
        
        Note: This test requires internet connection.
        """
        df = loader.download("AAPL", start="2024-01-01", end="2024-01-10")
        assert len(df) > 0
        assert set(df.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}

    def test_cache_after_download(self, loader):
        """Test that data is cached after download."""
        # First download
        df1 = loader.download("AAPL", start="2024-01-01", end="2024-01-10")
        
        # Verify cached
        assert loader.store.exists("AAPL")
        
        # Load from cache (should be instant)
        df2 = loader.load("AAPL")
        assert df1.shape == df2.shape

    def test_load_uses_cache(self, loader):
        """Test load() prefers cache over downloading."""
        # Download once
        df1 = loader.download("AAPL", start="2024-01-01", end="2024-01-10")
        
        # Load again (from cache)
        start_time = time.time()
        df2 = loader.load("AAPL", auto_download=False)
        elapsed = time.time() - start_time
        
        assert df1.shape == df2.shape
        assert elapsed < 0.1  # Should be very fast (< 100ms)

    def test_load_without_cache_fails(self, loader):
        """Test that load fails when data not cached and auto_download=False."""
        with pytest.raises(FileNotFoundError):
            loader.load("NONEXISTENT", auto_download=False)

    def test_download_overwrite(self, loader):
        """Test overwrite during download."""
        df1 = loader.download("AAPL", start="2024-01-01", end="2024-01-10")
        
        # Try to download again without overwrite (should fail)
        with pytest.raises(FileExistsError):
            loader.download("AAPL", start="2024-01-01", end="2024-01-10", overwrite=False)
        
        # Download with overwrite (should succeed)
        df2 = loader.download("AAPL", start="2024-01-01", end="2024-01-10", overwrite=True)
        assert df1.shape == df2.shape

    def test_data_validation_nan(self, loader):
        """Test validation detects NaN values."""
        # Create invalid DataFrame with NaN
        import pandas as pd
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        invalid_df = pl.from_pandas(
            pd.DataFrame({
                "timestamp": dates,
                "open": [100.0, 101.0, None, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000000.0, 1100000.0, 1200000.0, 1300000.0, 1400000.0],
            })
        )
        
        with pytest.raises(ValueError, match="NaN"):
            loader._validate_data(invalid_df, "TEST")

    def test_data_validation_high_low(self, loader):
        """Test validation detects High < Low."""
        import pandas as pd
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        invalid_df = pl.from_pandas(
            pd.DataFrame({
                "timestamp": dates,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 101.0, 104.0, 105.0],  # Row 2: high < low
                "low": [99.0, 100.0, 102.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000000.0, 1100000.0, 1200000.0, 1300000.0, 1400000.0],
            })
        )
        
        with pytest.raises(ValueError, match="High < Low"):
            loader._validate_data(invalid_df, "TEST")

    def test_data_validation_negative_volume(self, loader):
        """Test validation detects negative volume."""
        import pandas as pd
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        invalid_df = pl.from_pandas(
            pd.DataFrame({
                "timestamp": dates,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000000.0, -1100000.0, 1200000.0, 1300000.0, 1400000.0],
            })
        )
        
        with pytest.raises(ValueError, match="negative volume"):
            loader._validate_data(invalid_df, "TEST")

    def test_performance_cache_load(self, loader):
        """Test that cached load is < 100ms."""
        # Download once
        loader.download("AAPL", start="2024-01-01", end="2024-12-31")
        
        # Measure cache load time
        start_time = time.time()
        loader.load("AAPL", auto_download=False)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"\nCached load time: {elapsed:.2f}ms")
        assert elapsed < 100, f"Cache load took {elapsed:.2f}ms, expected < 100ms"

    @pytest.mark.slow
    def test_performance_uncached_download(self, loader):
        """Test that uncached download is < 500ms (may vary with network).
        
        Mark as slow test since network latency varies.
        """
        start_time = time.time()
        loader.download("AAPL", start="2024-01-01", end="2024-01-10")
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"\nUncached download time: {elapsed:.2f}ms")
        # Note: Download includes network latency; just measure to understand baseline


class TestDataStoreAbstraction:
    """Test DataStore interface abstraction."""

    def test_data_store_is_abstract(self):
        """Test that DataStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataStore()

    def test_parquet_store_implements_interface(self):
        """Test that ParquetDataStore implements DataStore interface."""
        assert isinstance(ParquetDataStore(), DataStore)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
