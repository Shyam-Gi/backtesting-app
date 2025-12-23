"""Momentum trading strategy.

Momentum strategy identifies assets that are trending upward.
Buy when return over a lookback period exceeds a threshold.
"""

import polars as pl
from backtesting_system.strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Momentum Trading Strategy.
    
    Buy when price has increased by more than threshold over lookback period.
    Sell when price has decreased by more than threshold.
    
    Parameters:
        lookback_period: Period to calculate momentum (default: 20 days)
        threshold: Percentage change threshold for signals (default: 0.02 = 2%)
    
    Example:
        strategy = MomentumStrategy(lookback_period=20, threshold=0.02)
        df = strategy.generate_signals(data)
    """

    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__(lookback_period=lookback_period, threshold=threshold)

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate momentum signals.
        
        Args:
            df: Polars DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum_pct and signal columns
        """
        df = df.with_columns([
            pl.col('close')
            .pct_change(n=self.lookback_period)
            .alias('momentum_pct')
        ])
        
        return df.with_columns([
            pl.when(pl.col('momentum_pct') > self.threshold)
            .then(1)
            .when(pl.col('momentum_pct') < -self.threshold)
            .then(-1)
            .otherwise(0)
            .alias('signal')
        ])