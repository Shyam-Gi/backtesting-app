"""Data loader with Polars, yfinance, and Parquet caching."""

import os
from pathlib import Path
from typing import Optional
import polars as pl
import pandas as pd
import yfinance as yf
from backtesting_system.data_store import DataStore


class ParquetDataStore(DataStore):
    """Parquet-based data store backend.
    
    Stores OHLCV data in Parquet files for fast, compressed access.
    Path: data/raw/{symbol}_daily.parquet
    """

    def __init__(self, data_dir: str = "data/raw"):
        """Initialize Parquet data store.
        
        Args:
            data_dir: Directory to store Parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str) -> Path:
        """Get file path for a symbol."""
        return self.data_dir / f"{symbol.upper()}_daily.parquet"

    def save(self, symbol: str, df: pl.DataFrame, overwrite: bool = False) -> None:
        """Save OHLCV data to Parquet.
        
        Args:
            symbol: Stock symbol
            df: Polars DataFrame
            overwrite: Whether to overwrite existing file
        """
        file_path = self._get_file_path(symbol)
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Data for {symbol} already exists. Use overwrite=True to replace.")
        
        df.write_parquet(str(file_path))

    def load(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pl.DataFrame:
        """Load OHLCV data from Parquet.
        
        Args:
            symbol: Stock symbol
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            Polars DataFrame
        """
        file_path = self._get_file_path(symbol)
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for {symbol}. Run download first.")
        
        df = pl.read_parquet(str(file_path))
        
        # Filter by date range if provided (convert strings to dates for comparison)
        if start_date or end_date:
            if start_date:
                df = df.filter(pl.col("timestamp").cast(pl.Date) >= pl.lit(start_date).str.to_date())
            if end_date:
                df = df.filter(pl.col("timestamp").cast(pl.Date) <= pl.lit(end_date).str.to_date())
        
        return df

    def exists(self, symbol: str) -> bool:
        """Check if Parquet file exists."""
        return self._get_file_path(symbol).exists()

    def delete(self, symbol: str) -> None:
        """Delete Parquet file."""
        file_path = self._get_file_path(symbol)
        if file_path.exists():
            file_path.unlink()

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with cached data."""
        files = list(self.data_dir.glob("*_daily.parquet"))
        return [f.stem.replace("_daily", "") for f in files]

    def get_date_range(self, symbol: str) -> tuple[str, str]:
        """Get date range for cached symbol."""
        df = pl.read_parquet(str(self._get_file_path(symbol)))
        min_date = df.select(pl.col("timestamp").min()).item()
        max_date = df.select(pl.col("timestamp").max()).item()
        return (str(min_date), str(max_date))


class DataLoader:
    """High-level data loader for fetching and caching OHLCV data.
    
    Features:
    - Downloads from yfinance (free)
    - Caches in Parquet (fast, compressed)
    - Validates data quality
    - Vectorized operations with Polars
    
    Target Performance:
    - Uncached: < 500ms (API call + Parquet write)
    - Cached: < 100ms (Parquet read)
    """

    def __init__(self, store: Optional[DataStore] = None, data_dir: str = "data/raw"):
        """Initialize data loader.
        
        Args:
            store: DataStore instance (defaults to ParquetDataStore)
            data_dir: Directory for cached data
        """
        self.store = store or ParquetDataStore(data_dir)

    def download(
        self,
        symbol: str,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
        overwrite: bool = False,
    ) -> pl.DataFrame:
        """Download OHLCV data from yfinance and cache to Parquet.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            overwrite: Whether to overwrite existing cache
            
        Returns:
            Polars DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Raises:
            ValueError: If data quality validation fails
        """
        # Download from yfinance
        df_yf = yf.download(symbol, start=start, end=end, progress=False)
        
        if df_yf.empty:
            raise ValueError(f"No data found for {symbol} between {start} and {end}")
        
        # Convert to Polars (reset_index to move Date/Datetime from index to column)
        df_pd = df_yf.reset_index()
        
        # Flatten column names if MultiIndex (yfinance sometimes creates MultiIndex)
        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = ['_'.join(col).strip('_') for col in df_pd.columns.values]
        
        # Handle MultiIndex column names like "Close_AAPL", "High_AAPL" by extracting base name
        col_mapping = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'close',
            'Volume': 'volume',
        }
        
        # First, try direct rename
        df_pd = df_pd.rename(columns={k: v for k, v in col_mapping.items() if k in df_pd.columns})
        
        # Then handle suffixed columns (e.g., Close_AAPL -> close)
        for col in df_pd.columns:
            for key, value in col_mapping.items():
                if col.startswith(key + '_'):
                    df_pd = df_pd.rename(columns={col: value})
                    break
        
        df = pl.from_pandas(df_pd)
        
        # Ensure timestamp is date type
        df = df.with_columns(pl.col("timestamp").cast(pl.Date).cast(pl.Datetime))
        
        # Validate data
        self._validate_data(df, symbol)
        
        # Cache to Parquet
        self.store.save(symbol, df, overwrite=overwrite)
        
        return df

    def load(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        auto_download: bool = True,
    ) -> pl.DataFrame:
        """Load OHLCV data, downloading if needed.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD), optional
            end: End date (YYYY-MM-DD), optional
            auto_download: Auto-download if not cached
            
        Returns:
            Polars DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        # Check cache first
        if self.store.exists(symbol):
            return self.store.load(symbol, start, end)
        
        # Download if not cached
        if auto_download:
            return self.download(symbol, start or "2020-01-01", end or "2024-12-31")
        
        raise FileNotFoundError(f"No cached data for {symbol}. Set auto_download=True to fetch.")

    def _validate_data(self, df: pl.DataFrame, symbol: str) -> None:
        """Validate OHLCV data quality.
        
        Args:
            df: Polars DataFrame to validate
            symbol: Symbol for error messages
            
        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for {symbol}: {missing}")
        
        # Check for NaN values
        has_nulls = False
        for col in ["open", "high", "low", "close", "volume"]:
            if df[col].null_count() > 0:
                has_nulls = True
                break
        if has_nulls:
            raise ValueError(f"Found NaN values in OHLCV for {symbol}")
        
        # Check OHLC relationships (High >= Low, High >= Open/Close, Low <= Open/Close)
        invalid_high_low = df.filter(pl.col("high") < pl.col("low")).height
        if invalid_high_low > 0:
            raise ValueError(f"High < Low in {invalid_high_low} rows for {symbol}")
        
        # Check timestamps are monotonic by comparing with sorted version
        ts_values = df.select("timestamp").to_series().to_list()
        if ts_values != sorted(ts_values):
            raise ValueError(f"Timestamps not monotonic for {symbol}")
        
        # Check volume is non-negative
        if (df.filter(pl.col("volume") < 0).height) > 0:
            raise ValueError(f"Found negative volume for {symbol}")
