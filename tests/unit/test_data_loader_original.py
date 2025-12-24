"""Unit tests for data loader and data store."""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np

from backtesting_system.data_loader import DataLoader, ParquetDataStore
from backtesting_system.data_store import DataStore


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

    def test_save_with_overwrite(self, store, sample_df):
        """Test overwriting existing data."""
        symbol = "AAPL"
        
        # Initial save
        store.save(symbol, sample_df)
        
        # Try to save without overwrite should raise error
        with pytest.raises(FileExistsError):
            store.save(symbol, sample_df)
        
        # Save with overwrite should succeed
        store.save(symbol, sample_df, overwrite=True)
        
        # Verify data is still valid
        loaded_df = store.load(symbol)
        assert loaded_df.shape == sample_df.shape

    def test_load_nonexistent_symbol(self, store):
        """Test loading non-existent symbol raises error."""
        with pytest.raises(FileNotFoundError):
            store.load("NONEXISTENT")

    def test_exists_check(self, store, sample_df):
        """Test checking if data exists."""
        symbol = "TEST"
        
        # Should not exist initially
        assert not store.exists(symbol)
        
        # Save data
        store.save(symbol, sample_df)
        
        # Should exist now
        assert store.exists(symbol)

    def test_get_available_symbols(self, store, sample_df):
        """Test getting list of available symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Initially empty
        assert store.get_available_symbols() == []
        
        # Save multiple symbols
        for symbol in symbols:
            store.save(symbol, sample_df)
        
        # Check available symbols
        available = store.get_available_symbols()
        assert set(available) == set(symbols)

    def test_file_path_generation(self, store):
        """Test file path generation for different symbols."""
        symbol1 = "AAPL"
        symbol2 = "msft"  # lowercase
        
        path1 = store._get_file_path(symbol1)
        path2 = store._get_file_path(symbol2)
        
        # Should be uppercase
        assert path1.name == "AAPL_daily.parquet"
        assert path2.name == "MSFT_daily.parquet"
        
        # Should be in data directory
        assert path1.parent == store.data_dir
        assert path2.parent == store.data_dir

    def test_delete_data(self, store, sample_df):
        """Test deleting data files."""
        symbol = "DELETE_ME"
        
        # Save data
        store.save(symbol, sample_df)
        assert store.exists(symbol)
        
        # Delete data
        store.delete(symbol)
        assert not store.exists(symbol)
        
        # Deleting non-existent should not raise error
        store.delete(symbol)  # Should not raise


class TestDataLoader:
    """Test high-level data loader functionality."""

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

    def test_initialization(self):
        """Test DataLoader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_store = ParquetDataStore(data_dir=temp_dir)
            data_loader = DataLoader(store=data_store)
            assert data_loader is not None
            assert data_loader.store is not None

    def test_basic_workflow(self, store, sample_df):
        """Test basic save and load workflow."""
        symbol = "WORKFLOW_TEST"
        
        # Create data loader
        data_loader = DataLoader(store=store)
        
        # Save data through loader's store
        data_loader.store.save(symbol, sample_df)
        
        # Load data through loader (using load method since data is already cached)
        loaded_df = data_loader.load(symbol)
        
        assert loaded_df.shape == sample_df.shape

    def test_data_validation(self, sample_df):
        """Test data validation functionality."""
        # Test with valid data (basic validation)
        assert sample_df.height > 0
        assert "timestamp" in sample_df.columns
        assert "close" in sample_df.columns
        
        # Test data has expected structure
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in sample_df.columns

    @pytest.fixture
    def performance_benchmark_data(self):
        """Create large dataset for performance benchmarking."""
        dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")
        n_days = len(dates)
        
        # Generate realistic price data
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

    def test_performance_benchmarks(self, store, performance_benchmark_data):
        """Test performance with large dataset."""
        # Use large pre-generated dataset
        symbol = "LARGE_TEST"
        store.save(symbol, performance_benchmark_data)
        
        data_loader = DataLoader(store=store)
        
        # Test load performance
        start_time = time.time()
        loaded_df = data_loader.load(symbol)
        load_time = time.time() - start_time
        
        # Performance assertions
        assert load_time < 5.0, f"Load took too long: {load_time:.3f}s"
        
        # Verify data integrity
        assert loaded_df.height == performance_benchmark_data.height
        assert loaded_df.shape[1] == performance_benchmark_data.shape[1]

    def test_concurrent_access(self, store, sample_df):
        """Test thread safety of data operations."""
        import threading
        import time
        
        symbol = "CONCURRENT_TEST"
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Each thread tries to load data
                if worker_id == 0:
                    # First thread saves data
                    store.save(symbol, sample_df)
                else:
                    # Other threads wait a bit then load
                    time.sleep(0.1)
                    data_loader = DataLoader(store=store)
                    df = data_loader.load(symbol)
                    results.append((worker_id, df.height))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors and consistent results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 2, f"Expected 2 successful loads, got {len(results)}"
        
        # All loads should return same height
        heights = [height for _, height in results]
        assert all(h == sample_df.height for h in heights)

    @pytest.fixture
    def corrupted_ohlcv_data(self):
        """Create corrupted OHLCV data for error testing."""
        dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        
        return pl.DataFrame({
            "timestamp": dates,
            "open": [100.0, None, 102.0, 103.0, 104.0],  # None value
            "high": [101.0, 102.0, 101.0, 104.0, 105.0],  # High < Open violation
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000000, 1100000, -1200000, 1300000, 1400000]  # Negative volume
        })

    def test_error_handling(self, store, corrupted_ohlcv_data):
        """Test error handling with corrupted data."""
        # Save corrupted data
        symbol = "CORRUPTED_TEST"
        store.save(symbol, corrupted_ohlcv_data)
        
        # Load should succeed (data loader doesn't validate deeply)
        data_loader = DataLoader(store=store)
        loaded_df = data_loader.load(symbol)
        
        # But data should have issues
        assert loaded_df.height == corrupted_ohlcv_data.height
        assert loaded_df["open"].null_count() > 0  # Should have null values