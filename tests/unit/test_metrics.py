"""Tests for metrics module.

Unit tests for performance metrics calculation and analysis.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from backtesting_system.metrics import MetricsCalculator, PerformanceMetrics
from backtesting_system.simulator import Trade


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    @pytest.fixture
    def sample_nav_history(self):
        """Create sample NAV history."""
        dates = []
        values = []
        
        start_date = datetime(2020, 1, 1)
        base_value = 100000
        
        for i in range(252):  # One trading year
            date = start_date + timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            
            # Simple growth with volatility
            daily_return = 0.0005 + np.random.normal(0, 0.01)  # 0.05% daily mean with volatility
            base_value *= (1 + daily_return)
            values.append(base_value)
        
        return pl.DataFrame({
            'timestamp': dates,
            'cash': [base_value * 0.3 for _ in values],  # 30% cash
            'position_value': [base_value * 0.7 for _ in values],  # 70% position
            'total_value': values,
            'position_quantity': [700 for _ in values],  # Constant position
            'unrealized_pnl': [0.0 for _ in values],
            'realized_pnl': [0.0 for _ in values],
            'total_pnl': [0.0 for _ in values]
        })
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        return [
            Trade('2020-01-15', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-02-20', 'SELL', 110.0, 100, 5.5, 1.0, 5.5),
            Trade('2020-03-10', 'BUY', 105.0, 150, 7.875, 1.575, 157.875),
            Trade('2020-04-15', 'SELL', 95.0, 150, 7.125, 1.425, 7.125),  # Losing trade
            Trade('2020-05-20', 'BUY', 98.0, 200, 9.8, 1.96, 196.96),
            Trade('2020-06-25', 'SELL', 108.0, 200, 10.8, 2.16, 10.8)
        ]
    
    def test_initialization(self):
        """Test calculator initialization."""
        calc = MetricsCalculator(risk_free_rate=0.03)
        assert calc.risk_free_rate == 0.03
    
    def test_empty_data_metrics(self):
        """Test metrics calculation with empty data."""
        calc = MetricsCalculator()
        empty_nav = pl.DataFrame(schema={
            'timestamp': pl.Utf8,
            'cash': pl.Float64,
            'position_value': pl.Float64,
            'total_value': pl.Float64,
            'position_quantity': pl.Int64,
            'unrealized_pnl': pl.Float64,
            'realized_pnl': pl.Float64,
            'total_pnl': pl.Float64
        })
        
        metrics = calc.calculate(empty_nav, [])
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
        assert metrics.cagr == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.trade_count == 0
    
    def test_basic_return_calculation(self, sample_nav_history):
        """Test basic return calculations."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, [])
        
        # Check that metrics are calculated (could be positive or negative due to randomness)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.cagr, float)
        assert isinstance(metrics.annualized_return, float)
        
        # Daily statistics should be reasonable
        assert abs(metrics.daily_return_mean) < 0.1  # Should not be extreme
        assert metrics.daily_return_std >= 0  # Should have some volatility (could be zero)
        
        # If we have a positive trend, returns should be positive
        if sample_nav_history['total_value'][-1] > sample_nav_history['total_value'][0]:
            assert metrics.total_return > 0
    
    def test_risk_metrics_calculation(self, sample_nav_history):
        """Test risk-related metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, [])
        
        # Risk metrics should be positive and reasonable
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
        assert metrics.max_drawdown <= 1  # Should not exceed 100%
        assert isinstance(metrics.calmar_ratio, float)  # Can be negative
        
        # Drawdown duration should be positive integer
        assert isinstance(metrics.max_drawdown_duration, int)
        assert metrics.max_drawdown_duration >= 0
    
    def test_risk_adjusted_returns(self, sample_nav_history):
        """Test risk-adjusted return calculations."""
        calc = MetricsCalculator(risk_free_rate=0.02)
        metrics = calc.calculate(sample_nav_history, [])
        
        # Risk-adjusted metrics should be reasonable
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        
        # For positive returns and reasonable risk
        if metrics.total_return > 0:
            assert metrics.sharpe_ratio > 0 or abs(metrics.sharpe_ratio) < 0.01
    
    def test_trade_statistics(self, sample_trades, sample_nav_history):
        """Test trade statistics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, sample_trades)
        
        # Trade count should match
        assert metrics.trade_count == len(sample_trades)
        
        # Win rate should be between 0 and 1
        assert 0 <= metrics.win_rate <= 1
        
        # Average win and loss should be reasonable
        assert metrics.avg_win >= 0 or metrics.avg_win == 0  # Can be zero if no wins
        assert metrics.avg_loss <= 0 or metrics.avg_loss == 0  # Can be zero if no losses
        
        # Profit factor should be positive or infinite
        assert metrics.profit_factor >= 0
        
        # Expectancy should be reasonable
        assert isinstance(metrics.expectancy, float)
    
    def test_efficiency_metrics(self, sample_nav_history, sample_trades):
        """Test efficiency and distribution metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, sample_trades)
        
        # Turnover should be non-negative
        assert metrics.turnover >= 0
        
        # VaR should be negative (loss) and reasonable
        assert metrics.var_95 <= 0
        
        # Skewness and kurtosis should be reasonable
        assert isinstance(metrics.skewness, float)
        assert isinstance(metrics.kurtosis, float)
    
    def test_benchmark_comparison(self, sample_nav_history, sample_trades):
        """Test benchmark comparison metrics."""
        # Create benchmark data
        benchmark_dates = sample_nav_history['timestamp'].to_list()
        benchmark_prices = [100 + i * 0.3 for i in range(len(benchmark_dates))]
        
        benchmark_data = pl.DataFrame({
            'timestamp': benchmark_dates,
            'open': benchmark_prices,
            'high': [p * 1.02 for p in benchmark_prices],
            'low': [p * 0.98 for p in benchmark_prices],
            'close': benchmark_prices,
            'volume': [1000000] * len(benchmark_prices)
        })
        
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, sample_trades, benchmark_data)
        
        # Benchmark-related metrics should be calculated
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.information_ratio, float)
    
    def test_metrics_with_losses(self):
        """Test metrics calculation with losing trades."""
        # Create losing NAV history
        dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        values = [100000, 98000, 96000, 94000, 92000]  # Declining portfolio
        
        nav_history = pl.DataFrame({
            'timestamp': dates,
            'cash': [46000] * len(values),
            'position_value': [46000] * len(values),
            'total_value': values,
            'position_quantity': [460] * len(values),
            'unrealized_pnl': [0.0] * len(values),
            'realized_pnl': [0.0] * len(values),
            'total_pnl': [0.0] * len(values)
        })
        
        calc = MetricsCalculator()
        metrics = calc.calculate(nav_history, [])
        
        # Should show negative returns
        assert metrics.total_return < 0
        assert metrics.cagr < 0
        
        # Risk metrics should still work
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
        
        # Sharpe ratio should be negative or low
        assert metrics.sharpe_ratio <= 0
    
    def test_metrics_consistency(self, sample_nav_history, sample_trades):
        """Test internal consistency of metrics."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, sample_trades)
        
        # Check relationships between metrics
        initial_value = sample_nav_history['total_value'][0]
        final_value = sample_nav_history['total_value'][-1]
        expected_total_return = (final_value - initial_value) / initial_value
        
        assert abs(metrics.total_return - expected_total_return) < 0.001
        
        # Check that risk-adjusted ratios are reasonable given inputs
        if metrics.volatility > 0:
            assert abs(metrics.sharpe_ratio - (metrics.annualized_return - 0.02) / metrics.volatility) < 0.01
    
    def test_metrics_with_no_trades(self, sample_nav_history):
        """Test metrics when no trades executed."""
        calc = MetricsCalculator()
        metrics = calc.calculate(sample_nav_history, [])
        
        # Trade statistics should be zero
        assert metrics.trade_count == 0
        assert metrics.win_rate == 0.0
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.expectancy == 0.0
        assert metrics.avg_holding_period == 0.0
        
        # Return and risk metrics should still work
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            annualized_return=0.13,
            daily_return_mean=0.0005,
            daily_return_std=0.01,
            volatility=0.16,
            max_drawdown=0.08,
            max_drawdown_duration=20,
            calmar_ratio=1.5,
            sharpe_ratio=0.8,
            sortino_ratio=1.2,
            information_ratio=0.3,
            beta=0.9,
            alpha=0.02,
            win_rate=0.6,
            avg_win=500.0,
            avg_loss=-300.0,
            profit_factor=1.8,
            expectancy=50.0,
            trade_count=25,
            avg_holding_period=15.5,
            turnover=0.5,
            var_95=-0.02,
            skewness=0.1,
            kurtosis=0.5
        )
        
        assert metrics.total_return == 0.15
        assert metrics.cagr == 0.12
        assert metrics.sharpe_ratio == 0.8
        assert metrics.win_rate == 0.6
        assert metrics.trade_count == 25
        assert isinstance(metrics.max_drawdown_duration, int)


class TestMetricsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_day_portfolio(self):
        """Test portfolio with only one day of data."""
        nav_history = pl.DataFrame({
            'timestamp': ['2020-01-01'],
            'cash': [50000],
            'position_value': [50000],
            'total_value': [100000],
            'position_quantity': [500],
            'unrealized_pnl': [0],
            'realized_pnl': [0],
            'total_pnl': [0]
        })
        
        calc = MetricsCalculator()
        metrics = calc.calculate(nav_history, [])
        
        # Should handle gracefully
        assert metrics.total_return == 0.0  # No change in one day
        assert isinstance(metrics.volatility, float)
    
    def test_constant_portfolio_value(self):
        """Test portfolio with constant value."""
        dates = ['2020-01-01', '2020-01-02', '2020-01-03']
        constant_value = 100000
        
        nav_history = pl.DataFrame({
            'timestamp': dates,
            'cash': [constant_value * 0.5] * len(dates),
            'position_value': [constant_value * 0.5] * len(dates),
            'total_value': [constant_value] * len(dates),
            'position_quantity': [500] * len(dates),
            'unrealized_pnl': [0.0] * len(dates),
            'realized_pnl': [0.0] * len(dates),
            'total_pnl': [0.0] * len(dates)
        })
        
        calc = MetricsCalculator()
        metrics = calc.calculate(nav_history, [])
        
        # Zero return and zero volatility
        assert metrics.total_return == 0.0
        assert metrics.daily_return_std == 0.0
        assert metrics.volatility == 0.0
        
        # Risk-adjusted ratios should be zero or undefined (handled as 0)
        assert metrics.sharpe_ratio == 0.0