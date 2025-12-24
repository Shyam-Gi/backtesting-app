"""Tests for accounting module.

Unit tests for portfolio accounting, NAV tracking, and P&L calculations.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from backtesting_system.accounting import Accounting, PortfolioSnapshot, Position
from backtesting_system.simulator import Trade, Simulator


class TestAccounting:
    """Test suite for Accounting class."""
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        return [
            Trade('2020-01-15', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-02-20', 'SELL', 110.0, 100, 5.5, 1.0, 5.5),
            Trade('2020-03-10', 'BUY', 105.0, 150, 7.875, 1.575, 157.875),
            Trade('2020-04-15', 'SELL', 115.0, 150, 8.625, 1.725, 8.625)
        ]
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = []
        prices = []
        
        start_date = datetime(2020, 1, 1)
        for i in range(120):  # 4 months of daily data
            date = start_date + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            # Simple price trend
            base_price = 100 + i * 0.5
            prices.append(base_price + np.sin(i * 0.1) * 5)
        
        return pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        })
    
    def test_initialization(self):
        """Test accounting initialization."""
        acct = Accounting(initial_cash=100000)
        
        assert acct.initial_cash == 100000
        assert acct.cash == 100000
        assert acct.position == 0
        assert acct.entry_price == 0.0
        assert acct.realized_pnl == 0.0
        assert acct.unrealized_pnl == 0.0
        assert len(acct.portfolio_history) == 0
        assert len(acct.trades_processed) == 0
    
    def test_process_no_trades(self, sample_price_data):
        """Test processing with no trades."""
        acct = Accounting(initial_cash=100000)
        acct.process_trades([], sample_price_data)
        
        # Should have portfolio history but no trades
        assert len(acct.portfolio_history) > 0
        assert len(acct.trades_processed) == 0
        assert acct.cash == 100000
        assert acct.position == 0
        
        # Check final portfolio value
        final_value = acct.get_final_portfolio_value()
        assert final_value == 100000  # No trades, cash unchanged
    
    def test_process_simple_trade(self, sample_trades, sample_price_data):
        """Test processing a simple buy-sell trade pair."""
        acct = Accounting(initial_cash=100000)
        
        # Process just first buy and sell
        simple_trades = sample_trades[:2]
        acct.process_trades(simple_trades, sample_price_data)
        
        assert len(acct.trades_processed) == 2
        assert acct.position == 0  # Should be flat after sell
        
        # Check realized P&L (accounting processes through portfolio history)
        # For profitable trades, P&L should be positive
        assert acct.realized_pnl > 0
    
    def test_total_return_percentage(self, sample_trades, sample_price_data):
        """Test total return is calculated as percentage."""
        acct = Accounting(initial_cash=100000)
        
        # Process trades
        acct.process_trades(sample_trades, sample_price_data)
        
        # Get total return - should be percentage format
        total_return = acct.get_total_return()
        
        # Return should be in percentage (not decimal)
        assert isinstance(total_return, float)
        
        # For profitable trades, return should be positive
        if acct.get_final_portfolio_value() > acct.initial_cash:
            assert total_return > 0
        
        # Check that return is properly scaled as percentage
        expected_decimal = (acct.get_final_portfolio_value() - acct.initial_cash) / acct.initial_cash
        expected_percentage = expected_decimal * 100
        assert abs(total_return - expected_percentage) < 0.01
    
    def test_total_return_no_trades(self, sample_price_data):
        """Test total return with no trades."""
        acct = Accounting(initial_cash=100000)
        acct.process_trades([], sample_price_data)
        
        # Should have zero return without trades
        total_return = acct.get_total_return()
        assert total_return == 0.0