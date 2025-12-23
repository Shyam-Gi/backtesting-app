"""Abstract data store interface for swappable backends."""

from abc import ABC, abstractmethod
from typing import Optional
import polars as pl


class DataStore(ABC):
    """Abstract base class for data storage backends.
    
    Supports multiple backends (Parquet, DuckDB, PostgreSQL) without changing
    the interface. Enables migration path: Parquet (V1) → DuckDB (V2) → PostgreSQL (V3).
    """

    @abstractmethod
    def save(self, symbol: str, df: pl.DataFrame, overwrite: bool = False) -> None:
        """Save OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            df: Polars DataFrame with columns [timestamp, open, high, low, close, volume]
            overwrite: Whether to overwrite existing data
        """
        pass

    @abstractmethod
    def load(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pl.DataFrame:
        """Load OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            
        Returns:
            Polars DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def exists(self, symbol: str) -> bool:
        """Check if data exists for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, symbol: str) -> None:
        """Delete data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Get list of all available symbols in store.
        
        Returns:
            List of symbol strings
        """
        pass

    @abstractmethod
    def get_date_range(self, symbol: str) -> tuple[str, str]:
        """Get available date range for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        pass
