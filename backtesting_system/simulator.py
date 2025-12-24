"""Vectorized execution simulator for backtesting.

Simulates realistic order execution with commission and slippage.
Uses vectorized Polars operations for speed.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade execution."""
    timestamp: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    commission: float
    slippage: float
    total_cost: float


class Simulator:
    """Vectorized backtesting simulator.
    
    Simulates order execution with realistic market frictions.
    Uses vectorized operations for performance.
    
    Parameters:
        initial_cash: Starting cash amount
        commission_bps: Commission in basis points (0.0001 = 1 bps = 0.01%)
        slippage_pct: Slippage as percentage of price
        position_size_pct: Percentage of available cash to use per trade
    
    Example:
        sim = Simulator(
            initial_cash=100000,
            commission_bps=5,      # 5 bps = 0.05%
            slippage_pct=0.001,   # 0.1% slippage
            position_size_pct=1.0   # Use 100% of available cash
        )
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_bps: float = 5.0,
        slippage_pct: float = 0.001,
        position_size_pct: float = 1.0
    ):
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        
        # State variables
        self.cash = initial_cash
        self.position = 0  # Number of shares held
        self.trades: List[Trade] = []
        
        # Commission and slippage calculations
        self.commission_rate = commission_bps / 10000.0  # Convert bps to decimal
        self.slippage_rate = slippage_pct

    def execute(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Execute trading signals through the data.
        
        Args:
            df: DataFrame with OHLCV data and 'signal' column from strategy
            
        Returns:
            Dictionary with trades, final portfolio value, and performance metrics
        """
        if 'signal' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' column from strategy")
        
# Create execution DataFrame with all needed columns
        exec_df = df.with_columns([
            pl.col('signal').alias('new_signal'),
            pl.col('signal').shift(1).alias('prev_signal')
        ]).fill_null(strategy='zero')
        
        # Find signal changes (entry/exit points) where signal actually changes
        exec_df = exec_df.with_columns([
            (pl.col('new_signal') != pl.col('prev_signal')).alias('signal_change')
        ])
        
        trade_points = exec_df.filter(
            (pl.col('signal_change') & (pl.col('new_signal') != 0))
        )
        
        # Execute trades vectorized
        trades = []
        for row in trade_points.iter_rows(named=True):
            trade = self._execute_trade(row)
            if trade:
                trades.append(trade)
                self.trades.append(trade)
        
        # Calculate final portfolio value
        final_price = df.select(pl.col('close')).to_series().to_list()[-1]
        portfolio_value = self.cash + (self.position * final_price)
        
        return {
            'trades': trades,
            'initial_cash': self.initial_cash,
            'final_cash': self.cash,
            'final_position': self.position,
            'final_portfolio_value': portfolio_value,
            'total_return': (portfolio_value - self.initial_cash) / self.initial_cash,
            'num_trades': len(trades),
            'total_commission': sum(t.commission for t in trades),
            'total_slippage': sum(t.slippage for t in trades),
            'execution_details': {
                'commission_bps': self.commission_bps,
                'slippage_pct': self.slippage_pct,
                'position_size_pct': self.position_size_pct
            }
        }

    def _execute_trade(self, row: Dict[str, Any]) -> Optional[Trade]:
        """Execute a single trade based on signal.
        
        Args:
            row: Row containing signal and price data
            
        Returns:
            Trade object or None if no trade executed
        """
        signal = row['signal']
        price = row['close']
        timestamp = str(row['timestamp'])
        
        if signal == 1 and self.position <= 0:
            # BUY signal - invest available cash
            available_cash = self.cash * self.position_size_pct
            trade_price = price * (1 + self.slippage_rate)  # Buy higher (slippage)
            max_shares = int(available_cash / trade_price)
            
            if max_shares > 0:
                commission = max_shares * trade_price * self.commission_rate
                total_cost = (max_shares * trade_price) + commission
                
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.position += max_shares
                    
                    return Trade(
                        timestamp=timestamp,
                        side='BUY',
                        price=trade_price,
                        quantity=max_shares,
                        commission=commission,
                        slippage=(trade_price - price) * max_shares,
                        total_cost=total_cost
                    )
        
        elif signal == -1 and self.position > 0:
            # SELL signal - sell all position
            trade_price = price * (1 - self.slippage_rate)  # Sell lower (slippage)
            proceeds = self.position * trade_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission
            
            self.cash += net_proceeds
            shares_sold = self.position
            self.position = 0
            
            return Trade(
                timestamp=timestamp,
                side='SELL',
                price=trade_price,
                quantity=shares_sold,
                commission=commission,
                slippage=(price - trade_price) * shares_sold,
                total_cost=commission
            )
        
        return None

    def reset(self) -> None:
        """Reset simulator state for new backtest."""
        self.cash = self.initial_cash
        self.position = 0
        self.trades = []

    def get_trades_df(self) -> pl.DataFrame:
        """Get trades as Polars DataFrame."""
        if not self.trades:
            return pl.DataFrame(schema={
                'timestamp': pl.Utf8,
                'side': pl.Utf8,
                'price': pl.Float64,
                'quantity': pl.Int64,
                'commission': pl.Float64,
                'slippage': pl.Float64,
                'total_cost': pl.Float64
            })
        
        return pl.DataFrame([
            {
                'timestamp': t.timestamp,
                'side': t.side,
                'price': t.price,
                'quantity': t.quantity,
                'commission': t.commission,
                'slippage': t.slippage,
                'total_cost': t.total_cost
            }
            for t in self.trades
        ])