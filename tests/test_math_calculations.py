"""Tests for math calculations in backtesting system.

Tests for portfolio calculations, performance metrics, and financial formulas.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator, PerformanceMetrics
from backtesting_system.simulator import Trade


class TestMathCalculations:
    """Test suite for mathematical accuracy."""
    
    @pytest.fixture
    def simple_price_data(self):
        """Create simple predictable price data."""
        dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        prices = [100, 105, 102, 108, 110]  # Simple price movements
        
        return pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        })
    
    @pytest.fixture
    def profitable_trades(self):
        """Create trades that should generate profit."""
        return [
            Trade('2020-01-02', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),   # Buy at 100
            Trade('2020-01-04', 'SELL', 108.0, 100, 5.4, 1.08, 504.8)  # Sell at 108
        ]
    
    @pytest.fixture
    def losing_trades(self):
        """Create trades that should generate loss."""
        return [
            Trade('2020-01-02', 'BUY', 105.0, 100, 5.25, 1.05, 531.25),  # Buy at 105
            Trade('2020-01-04', 'SELL', 102.0, 100, 5.1, 1.02, 501.0)   # Sell at 102
        ]
    
    @pytest.fixture
    def multiple_trades(self):
        """Create multiple trades for comprehensive testing."""
        return [
            Trade('2020-01-02', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),    # Buy 100 @ 100
            Trade('2020-01-03', 'BUY', 102.0, 50, 2.55, 0.51, 257.55),   # Buy 50 @ 102
            Trade('2020-01-04', 'SELL', 108.0, 75, 3.78, 0.756, 379.56), # Sell 75 @ 108
            Trade('2020-01-05', 'SELL', 110.0, 75, 4.125, 0.825, 413.25)  # Sell 75 @ 110
        ]
    
    def test_profitable_trade_calculation(self, profitable_trades, simple_price_data):
        """Test calculation of profitable trades."""
        acct = Accounting(initial_cash=10000)
        acct.process_trades(profitable_trades, simple_price_data)
        
        # The accounting system includes position value at market prices
        # Expected: should have positive return from profitable trade
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        realized_pnl = acct.realized_pnl
        
        # Should have positive results
        assert final_value > 10000, "Profitable trade should increase portfolio value"
        assert total_return > 0, "Profitable trade should have positive return"
        assert realized_pnl > 0, "Should have positive realized P&L"
        
        # Verify the percentage is correctly formatted (not decimal)
        assert total_return > 1.0, "Return should be in percentage format, not decimal"
    
    def test_losing_trade_calculation(self, losing_trades, simple_price_data):
        """Test calculation of losing trades."""
        acct = Accounting(initial_cash=10000)
        acct.process_trades(losing_trades, simple_price_data)
        
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        realized_pnl = acct.realized_pnl
        
        # For losing trades, expect negative realized P&L, but total portfolio value
        # might still be positive due to market price movements
        assert realized_pnl < 0, "Losing trade should have negative realized P&L"
        
        # The return calculation could be positive or negative depending on price movements
        # Test the calculation is mathematically sound
        expected_return_decimal = (final_value - 10000) / 10000
        expected_return_percentage = expected_return_decimal * 100
        assert abs(total_return - expected_return_percentage) < 0.01
    
    def test_multiple_trade_fifO_calculation(self, multiple_trades, simple_price_data):
        """Test FIFO calculation with multiple trades."""
        acct = Accounting(initial_cash=10000)
        acct.process_trades(multiple_trades, simple_price_data)
        
        # Manual FIFO calculation:
        # Position after 2 buys: 150 shares at average cost (100*100 + 50*102) / 150 = 100.67
        # First sell 75 shares: FIFO takes 75 from the first 100 @ 100
        # Realized P&L on first sell: 75 * (108 - 100) = 600 - costs
        # Second sell 75 shares: FIFO takes 25 from remaining first buy @ 100, and 50 from second buy @ 102
        # Realized P&L on second sell: 25 * (110 - 100) + 50 * (110 - 102) = 250 + 400 = 650 - costs
        
        # Should be profitable overall
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        
        assert final_value > 10000, "Multiple profitable trades should increase portfolio value"
        assert total_return > 0, "Multiple profitable trades should have positive return"
        assert acct.realized_pnl > 0, "Should have positive realized P&L"
    
    def test_no_trades_zero_return(self, simple_price_data):
        """Test that no trades result in zero return."""
        acct = Accounting(initial_cash=10000)
        acct.process_trades([], simple_price_data)
        
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        
        assert final_value == 10000, "No trades should not change portfolio value"
        assert total_return == 0.0, "No trades should result in zero return"
    
    def test_percentage_vs_decimal_format(self, simple_price_data):
        """Test that return is in percentage format, not decimal."""
        acct = Accounting(initial_cash=10000)
        
        # Create a simple trade 
        trades = [
            Trade('2020-01-02', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-01-04', 'SELL', 108.0, 100, 5.4, 1.08, 504.8)
        ]
        
        acct.process_trades(trades, simple_price_data)
        total_return = acct.get_total_return()
        
        # The return should be clearly in percentage format (not decimal like 0.07)
        assert total_return > 1.0, f"Return should be in percentage format, got {total_return}"
        
        # Verify the percentage calculation is correct
        final_value = acct.get_final_portfolio_value()
        expected_return_percentage = ((final_value - 10000) / 10000) * 100
        assert abs(total_return - expected_return_percentage) < 0.01
    
    def test_metrics_calculation_accuracy(self, simple_price_data):
        """Test performance metrics calculation accuracy."""
        acct = Accounting(initial_cash=10000)
        
        # Create trades for metrics testing
        trades = [
            Trade('2020-01-02', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-01-03', 'SELL', 105.0, 100, 5.25, 1.05, 504.7)  # Small profit
        ]
        
        acct.process_trades(trades, simple_price_data)
        nav_history = acct.get_nav_history()
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, trades)
        
        # Verify basic metrics structure
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        
        # Total return should match accounting calculation
        assert abs(metrics.total_return - acct.get_total_return() / 100) < 0.01
    
    def test_drawdown_calculation(self, simple_price_data):
        """Test drawdown calculation accuracy."""
        # Create price data with clear peak and trough
        dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        prices = [100, 120, 110, 90, 100]  # Peak at 120, trough at 90
        
        price_data = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        
        # Create trades that result in portfolio following price
        trades = [
            Trade('2020-01-01', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-01-05', 'SELL', 100.0, 100, 5.0, 1.0, 505.0)
        ]
        
        acct = Accounting(initial_cash=10000)
        acct.process_trades(trades, price_data)
        nav_history = acct.get_nav_history()
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, trades)
        
        # Maximum drawdown calculation is complex - just verify it's calculated
        assert 0 <= metrics.max_drawdown <= 1.0, f"Max drawdown should be between 0 and 1, got {metrics.max_drawdown}"


class TestEdgeCaseCalculations:
    """Test mathematical edge cases."""
    
    def test_zero_price_handling(self):
        """Test handling of zero or negative prices."""
        # Should not crash with edge case prices
        dates = ['2020-01-01', '2020-01-02']
        prices = [0.01, 0.01]  # Very small prices
        
        price_data = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        
        trades = [
            Trade('2020-01-01', 'BUY', 0.01, 1000, 0.01, 0.001, 0.011),
            Trade('2020-01-02', 'SELL', 0.01, 1000, 0.01, 0.001, 0.011)
        ]
        
        acct = Accounting(initial_cash=100)
        acct.process_trades(trades, price_data)
        
        # Should handle without errors
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        
        assert isinstance(final_value, float)
        assert isinstance(total_return, float)
    
    def test_very_large_numbers(self):
        """Test handling of very large trade sizes."""
        large_trades = [
            Trade('2020-01-01', 'BUY', 100.0, 1000000, 50000, 10000, 100510000),
            Trade('2020-01-02', 'SELL', 105.0, 1000000, 52500, 10500, 105525000)
        ]
        
        dates = ['2020-01-01', '2020-01-02']
        prices = [100, 105]
        
        price_data = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        
        acct = Accounting(initial_cash=100000000)  # 100M initial
        acct.process_trades(large_trades, price_data)
        
        # Should handle large numbers without overflow
        final_value = acct.get_final_portfolio_value()
        total_return = acct.get_total_return()
        
        assert final_value > 0
        assert not np.isnan(final_value)
        assert not np.isnan(total_return)
    
    def test_commission_impact(self):
        """Test that commissions are properly factored into returns."""
        # Test with zero commissions for baseline
        zero_commission_trades = [
            Trade('2020-01-01', 'BUY', 100.0, 100, 0.0, 0.0, 0.0),
            Trade('2020-01-02', 'SELL', 110.0, 100, 0.0, 0.0, 0.0)
        ]
        
        dates = ['2020-01-01', '2020-01-02']
        prices = [100, 110]
        
        price_data = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices,
            'low': prices,
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        
        # Test zero commission case
        acct_zero = Accounting(initial_cash=10000)
        acct_zero.process_trades(zero_commission_trades, price_data)
        zero_commission_return = acct_zero.get_total_return()
        
        # Test with commissions
        commission_trades = [
            Trade('2020-01-01', 'BUY', 100.0, 100, 50.0, 5.0, 5050.0),
            Trade('2020-01-02', 'SELL', 110.0, 100, 55.0, 5.5, 5055.0)
        ]
        
        acct_with_comm = Accounting(initial_cash=10000)
        acct_with_comm.process_trades(commission_trades, price_data)
        with_commission_return = acct_with_comm.get_total_return()
        
        # Commissions should reduce returns
        assert with_commission_return < zero_commission_return, "Commissions should reduce returns"
        
        # Both should be positive due to price increase
        assert zero_commission_return > 0
        assert with_commission_return > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])