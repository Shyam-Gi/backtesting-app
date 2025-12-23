"""Abstract strategy base class for pluggable trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import polars as pl


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.
    
    Strategies must inherit from this class and implement generate_signals().
    
    Design Principles:
    - Vectorized: Use Polars expressions, no for-loops
    - Stateless: Pure function (same input -> same output)
    - Configurable: Parameters passed in __init__
    
    Signal Convention:
    - 1: BUY signal
    - -1: SELL signal
    - 0: HOLD/EXIT signal
    
    Example:
        class SMAStrategy(BaseStrategy):
            def __init__(self, fast_period=10, slow_period=50):
                self.fast_period = fast_period
                self.slow_period = slow_period
            
            def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
                df = df.with_columns([
                    pl.col('close').rolling(self.fast_period).mean().alias('fast_ma'),
                    pl.col('close').rolling(self.slow_period).mean().alias('slow_ma')
                ])
                
                return df.with_columns([
                    pl.when(pl.col('fast_ma') > pl.col('slow_ma'))
                    .then(1)
                    .when(pl.col('fast_ma') < pl.col('slow_ma'))
                    .then(-1)
                    .otherwise(0)
                    .alias('signal')
                ])
    """

    def __init__(self, **kwargs):
        """Initialize strategy with parameters.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        self.params = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from OHLCV data.
        
        This method must be implemented by all strategy subclasses.
        
        Args:
            df: Polars DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            Polars DataFrame with original columns plus a 'signal' column.
            Signal values: 1 (BUY), -1 (SELL), 0 (HOLD)
            
        Note:
            - Must use vectorized Polars expressions (no for-loops)
            - Should not modify original DataFrame (Polars is immutable)
            - Must return a DataFrame with 'signal' column
        """
        pass

    def validate_signals(self, df: pl.DataFrame) -> bool:
        """Validate generated signals.
        
        Args:
            df: DataFrame with 'signal' column
            
        Returns:
            True if signals are valid
            
        Raises:
            ValueError: If signals are invalid
        """
        if 'signal' not in df.columns:
            raise ValueError("Signal column missing from DataFrame")
        
        unique_signals = df.select(pl.col('signal').unique()).to_series().to_list()
        valid_signals = {1, -1, 0, None}
        
        for sig in unique_signals:
            if sig not in valid_signals and not (isinstance(sig, float) and sig != sig):
                raise ValueError(f"Invalid signal value: {sig}")
        
        return True

    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return self.params.copy()

    def get_name(self) -> str:
        """Get strategy class name.
        
        Returns:
            Strategy class name
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """String representation of strategy."""
        params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"{self.get_name()}({params_str})"
