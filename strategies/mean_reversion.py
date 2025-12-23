"""Mean Reversion trading strategy.

Mean reversion strategy identifies overbought/oversold conditions.
Buy when price deviates significantly below moving average.
Sell when price reverts to mean or above.
"""

import polars as pl
from backtesting_system.strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Trading Strategy.
    
    Buy when price is below moving average by threshold percentage.
    Sell when price is at or above moving average.
    
    Parameters:
        ma_period: Period for moving average (default: 20)
        threshold: Deviation threshold as percentage (default: 0.05 = 5%)
    
    Example:
        strategy = MeanReversionStrategy(ma_period=20, threshold=0.05)
        df = strategy.generate_signals(data)
    """

    def __init__(self, ma_period: int = 20, threshold: float = 0.05):
        super().__init__(ma_period=ma_period, threshold=threshold)

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate mean reversion signals.
        
        Args:
            df: Polars DataFrame with OHLCV data
            
        Returns:
            DataFrame with ma, distance, and signal columns
        """
        df = df.with_columns([
            pl.col('close')
            .rolling_mean(window_size=self.ma_period)
            .alias('ma')
        ])
        
        df = df.with_columns([
            ((pl.col('close') - pl.col('ma')) / pl.col('ma')).alias('distance')
        ])
        
        return df.with_columns([
            pl.when(pl.col('distance') < -self.threshold)
            .then(1)  # BUY when too far below MA
            .when(pl.col('distance') >= 0)
            .then(-1)  # SELL when at or above MA
            .otherwise(0)
            .alias('signal')
        ])