"""Portfolio accounting for backtesting.

Tracks NAV, positions, realized/unrealized P&L, and transaction history.
Uses vectorized Polars operations for performance.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .simulator import Trade


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PortfolioSnapshot:
    """Daily portfolio state."""
    timestamp: str
    cash: float
    position_value: float
    total_value: float
    position_quantity: int
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float


class Accounting:
    """Portfolio accounting system for backtesting.
    
    Tracks daily NAV, positions, P&L, and transaction history.
    Uses vectorized operations for performance.
    
    Parameters:
        initial_cash: Starting cash amount
        
    Example:
        acct = Accounting(initial_cash=100000)
        acct.process_trades(trades, price_data)
        nav_history = acct.get_nav_history()
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0  # Number of shares held
        self.entry_price = 0.0
        self.entry_date = None
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # History
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.trades_processed: List[Trade] = []
        
        # Position tracking for multiple symbols (future use)
        self.positions: Dict[str, Position] = {}
        
    def process_trades(self, trades: List[Trade], price_data: pl.DataFrame) -> None:
        """Process trades and update portfolio state.
        
        Args:
            trades: List of executed trades
            price_data: OHLCV data for position valuation
        """
        if not trades:
            # No trades, just track portfolio value over time
            self._calculate_portfolio_history(price_data)
            return
            
        # Sort trades by timestamp
        trades_sorted = sorted(trades, key=lambda t: t.timestamp)
        self.trades_processed = trades_sorted
        
        # Process each trade
        for trade in trades_sorted:
            self._process_single_trade(trade)
            
        # Calculate portfolio history after all trades
        self._calculate_portfolio_history(price_data)
        
    def _process_single_trade(self, trade: Trade) -> None:
        """Process a single trade and update accounting state."""
        if trade.side == 'BUY':
            # Update position and cash
            old_quantity = self.position
            old_cost_basis = old_quantity * self.entry_price if old_quantity > 0 else 0.0
            
            self.position += trade.quantity
            self.cash -= trade.total_cost
            
            # Calculate new entry price (weighted average)
            new_cost_basis = trade.quantity * trade.price
            total_cost_basis = old_cost_basis + new_cost_basis
            self.entry_price = total_cost_basis / self.position if self.position > 0 else 0.0
            self.entry_date = trade.timestamp
            
        elif trade.side == 'SELL':
            # Calculate realized P&L
            if self.position > 0 and self.entry_price > 0:
                proceeds_per_share = trade.price
                realized_pnl_per_share = proceeds_per_share - self.entry_price
                trade_realized_pnl = realized_pnl_per_share * trade.quantity
                
                self.realized_pnl += trade_realized_pnl
                
            # Update position and cash
            self.cash += (trade.quantity * trade.price) - trade.commission
            self.position -= trade.quantity
            
            # Reset entry data if no position left
            if self.position <= 0:
                self.position = 0
                self.entry_price = 0.0
                self.entry_date = None
                
    def _calculate_portfolio_history(self, price_data: pl.DataFrame) -> None:
        """Calculate daily portfolio values and P&L."""
        # Create price lookup dict
        price_lookup = {
            str(row['timestamp']): row['close'] 
            for row in price_data.select(['timestamp', 'close']).iter_rows(named=True)
        }
        
        # Get all unique timestamps from price data
        timestamps = price_data.select('timestamp').to_series().to_list()
        
        # Track portfolio state over time
        current_cash = self.initial_cash
        current_position = 0
        current_entry_price = 0.0
        current_realized_pnl = 0.0
        
        # Process trades in chronological order
        trades_by_date = {}
        for trade in self.trades_processed:
            if trade.timestamp not in trades_by_date:
                trades_by_date[trade.timestamp] = []
            trades_by_date[trade.timestamp].append(trade)
        
        # Calculate portfolio state for each timestamp
        for timestamp in timestamps:
            ts_str = str(timestamp)
            
            # Process any trades at this timestamp
            if ts_str in trades_by_date:
                for trade in trades_by_date[ts_str]:
                    if trade.side == 'BUY':
                        old_quantity = current_position
                        old_cost = old_quantity * current_entry_price if old_quantity > 0 else 0.0
                        
                        current_position += trade.quantity
                        current_cash -= trade.total_cost
                        
                        new_cost = trade.quantity * trade.price
                        total_cost = old_cost + new_cost
                        current_entry_price = total_cost / current_position if current_position > 0 else 0.0
                        
                    elif trade.side == 'SELL' and current_position > 0:
                        proceeds_per_share = trade.price
                        realized_pnl_per_share = proceeds_per_share - current_entry_price
                        trade_realized_pnl = realized_pnl_per_share * trade.quantity
                        
                        current_realized_pnl += trade_realized_pnl
                        current_cash += (trade.quantity * trade.price) - trade.commission
                        current_position -= trade.quantity
                        
                        if current_position <= 0:
                            current_position = 0
                            current_entry_price = 0.0
            
            # Calculate portfolio value at this timestamp
            current_price = price_lookup.get(ts_str, 0.0)
            position_value = current_position * current_price
            total_value = current_cash + position_value
            
            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            if current_position > 0 and current_entry_price > 0:
                unrealized_pnl = (current_price - current_entry_price) * current_position
            
            total_pnl = current_realized_pnl + unrealized_pnl
            
            # Create portfolio snapshot
            snapshot = PortfolioSnapshot(
                timestamp=ts_str,
                cash=current_cash,
                position_value=position_value,
                total_value=total_value,
                position_quantity=current_position,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=current_realized_pnl,
                total_pnl=total_pnl
            )
            
            self.portfolio_history.append(snapshot)
            
        # Update final state
        self.cash = current_cash
        self.position = current_position
        self.entry_price = current_entry_price
        self.realized_pnl = current_realized_pnl
        
    def get_nav_history(self) -> pl.DataFrame:
        """Get NAV history as Polars DataFrame."""
        if not self.portfolio_history:
            return pl.DataFrame(schema={
                'timestamp': pl.Utf8,
                'cash': pl.Float64,
                'position_value': pl.Float64,
                'total_value': pl.Float64,
                'position_quantity': pl.Int64,
                'unrealized_pnl': pl.Float64,
                'realized_pnl': pl.Float64,
                'total_pnl': pl.Float64
            })
        
        return pl.DataFrame([
            {
                'timestamp': snap.timestamp,
                'cash': snap.cash,
                'position_value': snap.position_value,
                'total_value': snap.total_value,
                'position_quantity': snap.position_quantity,
                'unrealized_pnl': snap.unrealized_pnl,
                'realized_pnl': snap.realized_pnl,
                'total_pnl': snap.total_pnl
            }
            for snap in self.portfolio_history
        ])
        
    def get_final_portfolio_value(self) -> float:
        """Get final portfolio value."""
        if not self.portfolio_history:
            return self.initial_cash
        return self.portfolio_history[-1].total_value
        
    def get_total_return(self) -> float:
        """Get total return as percentage."""
        final_value = self.get_final_portfolio_value()
        return (final_value - self.initial_cash) / self.initial_cash
        
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
        
    def get_trade_count(self) -> int:
        """Get number of trades processed."""
        return len(self.trades_processed)
        
    def get_win_rate(self) -> float:
        """Calculate win rate from profitable trades."""
        if not self.trades_processed:
            return 0.0
            
        profitable_trades = 0
        total_closed_trades = 0
        
        # Track positions for P&L calculation
        position_quantity = 0
        position_cost = 0.0
        
        for trade in self.trades_processed:
            if trade.side == 'BUY':
                position_quantity += trade.quantity
                position_cost += trade.quantity * trade.price
                
            elif trade.side == 'SELL' and position_quantity > 0:
                # Calculate P&L for this closed position
                proceeds = trade.quantity * trade.price
                cost_basis = (trade.quantity / position_quantity) * position_cost
                pnl = proceeds - cost_basis
                
                if pnl > 0:
                    profitable_trades += 1
                    
                total_closed_trades += 1
                
                # Update position tracking
                sold_ratio = trade.quantity / position_quantity
                position_cost *= (1 - sold_ratio)
                position_quantity -= trade.quantity
                
        return profitable_trades / total_closed_trades if total_closed_trades > 0 else 0.0
        
    def reset(self) -> None:
        """Reset accounting state for new backtest."""
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.entry_date = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.portfolio_history = []
        self.trades_processed = []
        self.positions = {}