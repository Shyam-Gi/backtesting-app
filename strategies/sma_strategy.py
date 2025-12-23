"""SMA Crossover trading strategy.

Simple Moving Average (SMA) crossover is a classic trend-following strategy.
When fast MA crosses above slow MA, it's a bullish signal (BUY).
When fast MA crosses below slow MA, it's a bearish signal (SELL).
"""

import polars as pl
from backtesting_system.strategy import BaseStrategy


class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy.
    
    Buy when fast MA > slow MA, Sell when fast MA < slow MA.
    
    Parameters:
        fast_period: Period for fast moving average (default: 10)
        slow_period: Period for slow moving average (default: 50)
    
    Example:
        strategy = SMAStrategy(fast_period=10, slow_period=50)
        df = strategy.generate_signals(data)
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        super().__init__(fast_period=fast_period, slow_period=slow_period)

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate SMA crossover signals.
        
        Args:
            df: Polars DataFrame with OHLCV data
            
        Returns:
            DataFrame with fast_ma, slow_ma, and signal columns
        """
        # Calculate moving averages using rolling windows
        df = df.with_columns([
            pl.col('close')
            .rolling_mean(window_size=self.fast_period)
            .alias('fast_ma'),
            
            pl.col('close')
            .rolling_mean(window_size=self.slow_period)
            .alias('slow_ma')
        ])
        
        # Generate signals based on crossover
        return df.with_columns([
            pl.when(pl.col('fast_ma') > pl.col('slow_ma'))
            .then(1)
            .when(pl.col('fast_ma') < pl.col('slow_ma'))
            .then(-1)
            .otherwise(0)
            .alias('signal')
        ])