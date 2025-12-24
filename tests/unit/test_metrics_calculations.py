"""Unit tests for metrics calculations and financial formulas."""

import pytest
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator, PerformanceMetrics
from backtesting_system.simulator import Trade


class TestMetricsCalculations:
    """Test suite for mathematical accuracy of performance metrics."""

    @pytest.fixture
    def small_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        return pl.DataFrame({
            'timestamp': dates,
            'open': [100 + i * 0.1 for i in range(10)],
            'high': [101 + i * 0.1 for i in range(10)],
            'low': [99 + i * 0.1 for i in range(10)],
            'close': [100.5 + i * 0.1 for i in range(10)],
            'volume': [1000000] * 10
        })

    @pytest.fixture
    def performance_benchmark_data(self):
        """Create large dataset for performance benchmarking."""
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        return pl.DataFrame({
            'timestamp': dates,
            'open': [100 + i * 0.01 for i in range(1000)],
            'high': [101 + i * 0.01 for i in range(1000)],
            'low': [99 + i * 0.01 for i in range(1000)],
            'close': [100.5 + i * 0.01 for i in range(1000)],
            'volume': [1000000] * 1000
        })

    def test_basic_metrics_structure(self, small_ohlcv_data):
        """Test that basic metrics calculation returns proper structure."""
        # Create simple NAV history
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'cash': [100000] * 10,
            'position_value': [0] * 10,
            'total_value': [100000, 102000, 101000, 105000, 104000, 
                           106000, 103000, 107000, 105000, 108000]
        })
        
        # Calculate metrics
        calculator = MetricsCalculator(risk_free_rate=0.02)
        metrics = calculator.calculate(nav_history, [])
        
        # Verify metrics is a PerformanceMetrics object
        assert isinstance(metrics, PerformanceMetrics)
        
        # Check all required fields are present
        required_fields = [
            'total_return', 'cagr', 'volatility', 'max_drawdown',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
            'trade_count', 'alpha', 'beta'
        ]
        
        for field in required_fields:
            assert hasattr(metrics, field), f"Missing field: {field}"
            assert getattr(metrics, field) is not None, f"Field {field} is None"

    def test_total_return_calculation(self):
        """Test total return calculation with known inputs."""
        # Create NAV history with known characteristics
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=6, freq='D'),
            'cash': [100000, 95000, 95000, 102000, 102000, 105000],
            'position_value': [0, 5000, 5200, 0, 0, 0],
            'total_value': [100000, 100000, 100200, 102000, 102000, 105000]
        })
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, [])
        
        # Expected total return: (105000 - 100000) / 100000 = 5%
        expected_return = 0.05
        assert abs(metrics.total_return - expected_return) < 0.001, \
            f"Expected {expected_return:.3f}, got {metrics.total_return:.3f}"

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation with different scenarios."""
        # Create steady profitable NAV history
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D'),
            'cash': [100000, 100000, 102000, 102000, 104000],
            'position_value': [0, 2000, 0, 3000, 0],
            'total_value': [100000, 102000, 102000, 105000, 104000]
        })
        
        calculator = MetricsCalculator(risk_free_rate=0.0)  # Zero risk-free rate
        metrics = calculator.calculate(nav_history, [])
        
        # Sharpe should be positive for profitable strategy
        assert metrics.sharpe_ratio > 0, f"Expected positive Sharpe, got {metrics.sharpe_ratio}"
        
        # Should be reasonable (< 10 for this small dataset)
        assert abs(metrics.sharpe_ratio) < 10, f"Suspicious Sharpe ratio: {metrics.sharpe_ratio}"

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create NAV history with known drawdown pattern
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=7, freq='D'),
            'cash': [100000, 105000, 95000, 100000, 94000, 110000, 110000],
            'position_value': [0, 0, 10000, 0, 15000, 0, 0],
            'total_value': [100000, 105000, 105000, 100000, 109000, 110000, 110000]
        })
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, [])
        
        # Max drawdown should be between 0 and 1 (percentage)
        assert 0 <= metrics.max_drawdown <= 1.0, f"Max drawdown should be between 0 and 1, got {metrics.max_drawdown}"

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        # Create NAV history with known volatility
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=6, freq='D'),
            'cash': [100000, 102000, 98000, 101000, 99000, 103000],
            'position_value': [0, 0, 3000, 0, 2000, 0],
            'total_value': [100000, 102000, 101000, 101000, 101000, 103000]
        })
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, [])
        
        # Volatility should be positive
        assert metrics.volatility >= 0, f"Volatility should be positive, got {metrics.volatility}"
        
        # Should be reasonable (< 100% annualized)
        assert metrics.volatility < 2.0, f"Suspicious volatility: {metrics.volatility}"

    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        calculator = MetricsCalculator()
        
        # Test with empty data
        empty_nav = pl.DataFrame({
            'timestamp': [],
            'cash': [],
            'position_value': [],
            'total_value': []
        })
        
        metrics = calculator.calculate(empty_nav, [])
        
        # Should handle gracefully
        assert metrics.total_return == 0.0
        assert metrics.trade_count == 0
        
        # Test with single data point
        single_nav = pl.DataFrame({
            'timestamp': ['2024-01-01'],
            'cash': [100000],
            'position_value': [0],
            'total_value': [100000]
        })
        
        metrics_single = calculator.calculate(single_nav, [])
        
        # Single data point should have zero return and volatility
        assert metrics_single.total_return == 0.0
        assert metrics_single.volatility == 0.0

    def test_with_actual_trades(self, small_ohlcv_data):
        """Test metrics calculation with actual trades from simulator."""
        # Create a simple strategy and simulate trades
        from strategies.sma_strategy import SMAStrategy
        strategy = SMAStrategy(fast_period=5, slow_period=10)
        
        from backtesting_system.simulator import Simulator
        simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
        
        # Generate signals and execute
        signals = strategy.generate_signals(small_ohlcv_data)
        result = simulator.execute(signals)
        
        # Create accounting and get NAV history
        accounting = Accounting(initial_cash=100000)
        accounting.process_trades(result['trades'], signals)
        nav_history = accounting.get_nav_history()
        
        # Calculate metrics
        calculator = MetricsCalculator(risk_free_rate=0.02)
        metrics = calculator.calculate(nav_history, result['trades'])
        
        # Should have meaningful metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.trade_count >= 0
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)

    @pytest.mark.slow
    def test_large_dataset_performance(self, performance_benchmark_data, perf_tracker):
        """Test metrics calculation performance with large dataset."""
        # Create NAV history from large dataset
        nav_history = pl.DataFrame({
            'timestamp': performance_benchmark_data['timestamp'],
            'cash': [100000] * len(performance_benchmark_data),
            'position_value': [0] * len(performance_benchmark_data),
            'total_value': [100000] * len(performance_benchmark_data)
        })
        
        calculator = MetricsCalculator()
        
        # Benchmark calculation
        perf_tracker.start_timer("large_dataset_metrics")
        metrics = calculator.calculate(nav_history, [])
        calculation_time = perf_tracker.end_timer("large_dataset_metrics")
        
        # Should be fast (< 1 second for large dataset)
        assert calculation_time < 1.0, \
            f"Metrics calculation too slow: {calculation_time:.3f}s"
        
        # Should have all standard metrics
        required_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'volatility',
            'win_rate', 'profit_factor', 'calmar_ratio'
        ]
        for metric in required_metrics:
            assert hasattr(metrics, metric)
            assert getattr(metrics, metric) is not None

    def test_risk_adjusted_ratios(self):
        """Test various risk-adjusted ratios."""
        # Create NAV history with known risk characteristics
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'cash': [100000] * 10,
            'position_value': [0] * 10,
            'total_value': [100000, 102000, 101000, 105000, 103000,
                           106000, 104000, 108000, 107000, 110000]
        })
        
        calculator = MetricsCalculator(risk_free_rate=0.02)
        metrics = calculator.calculate(nav_history, [])
        
        # All risk-adjusted ratios should be calculated
        risk_adjusted_ratios = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        
        for ratio in risk_adjusted_ratios:
            ratio_value = getattr(metrics, ratio)
            assert ratio_value is not None, f"Risk ratio {ratio} is None"
            
            # Should be finite (not NaN or infinity)
            assert np.isfinite(ratio_value), f"Risk ratio {ratio} is not finite: {ratio_value}"

    def test_cagr_calculation(self):
        """Test Compound Annual Growth Rate calculation."""
        # Create 1 year of data with known growth
        nav_history = pl.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=252, freq='B'),  # Trading days
            'cash': [100000] * 252,
            'position_value': [0] * 252,
            'total_value': [100000 + i * 100 for i in range(252)]  # Linear growth
        })
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, [])
        
        # Should have positive CAGR
        assert metrics.cagr > 0, f"CAGR should be positive, got {metrics.cagr}"
        
        # Should be reasonable
        assert metrics.cagr < 2.0, f"Suspicious CAGR: {metrics.cagr}"