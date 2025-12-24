"""Integration tests for end-to-end backtest pipeline."""

import pytest
import polars as pl
from pathlib import Path
import tempfile
import shutil

from backtesting_system.data_loader import DataLoader, ParquetDataStore
from backtesting_system.simulator import Simulator
from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy

from tests.utils import generate_ohlcv_with_trend, validate_ohlcv_data


class TestEndToEndBacktest:
    """Test complete backtest pipeline integration."""

    def test_full_pipeline_single_strategy(self, temp_dir, perf_tracker):
        """Test complete pipeline from data to metrics."""
        # Setup
        data_store = ParquetDataStore(data_dir=temp_dir)
        data_loader = DataLoader(store=data_store)
        strategy = SMAStrategy(fast_period=10, slow_period=50)
        simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
        accounting = Accounting(initial_cash=100000)
        calculator = MetricsCalculator(risk_free_rate=0.02)
        
        # Generate test data
        perf_tracker.start_timer("data_generation")
        test_data = generate_ohlcv_with_trend(
            start_date="2023-01-01", 
            end_date="2023-12-31",
            trend=0.0005,
            volatility=0.02,
            seed=42
        )
        perf_tracker.end_timer("data_generation")
        
        # Load data (simulating real workflow)
        symbol = "TEST_STOCK"
        data_store.save(symbol, test_data)
        
        perf_tracker.start_timer("data_loading")
        df = data_loader.download(symbol, start="2023-01-01", end="2023-12-31")
        perf_tracker.end_timer("data_loading")
        
        # Generate signals
        perf_tracker.start_timer("signal_generation")
        signals_df = strategy.generate_signals(df)
        perf_tracker.end_timer("signal_generation")
        
        # Execute trades
        perf_tracker.start_timer("trade_execution")
        result = simulator.execute(signals_df)
        perf_tracker.end_timer("trade_execution")
        
        # Process accounting
        perf_tracker.start_timer("accounting")
        accounting.process_trades(result['trades'], signals_df)
        perf_tracker.end_timer("accounting")
        
        # Calculate metrics
        perf_tracker.start_timer("metrics_calculation")
        nav_history = accounting.get_nav_history()
        metrics = calculator.calculate(nav_history, result['trades'])
        perf_tracker.end_timer("metrics_calculation")
        
        # Assertions
        assert result['num_trades'] > 0, "Should have executed trades"
        assert len(nav_history) > 0, "Should have NAV history"
        assert metrics['total_return'] is not None, "Should calculate total return"
        assert metrics['sharpe_ratio'] is not None, "Should calculate Sharpe ratio"
        
        # Performance targets
        total_time = sum(perf_tracker.timings[t]['duration'] for t in perf_tracker.timings)
        assert total_time < 2.0, f"Pipeline too slow: {total_time:.3f}s"

    def test_multiple_strategies_comparison(self, temp_dir):
        """Test running multiple strategies and comparing results."""
        # Setup
        data_store = ParquetDataStore(data_dir=temp_dir)
        data_loader = DataLoader(store=data_store)
        
        strategies = {
            "SMA_10_50": SMAStrategy(fast_period=10, slow_period=50),
            "SMA_20_100": SMAStrategy(fast_period=20, slow_period=100),
            "Momentum_20": MomentumStrategy(lookback_period=20, threshold=0.02)
        }
        
        # Generate and save test data
        test_data = generate_ohlcv_with_trend(
            start_date="2023-01-01", 
            end_date="2023-06-30",
            trend=0.0003,
            volatility=0.015,
            seed=123
        )
        symbol = "COMPARE_TEST"
        data_store.save(symbol, test_data)
        df = data_loader.download(symbol)
        
        results = {}
        
        for name, strategy in strategies.items():
            # Run backtest for each strategy
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            accounting = Accounting(initial_cash=100000)
            calculator = MetricsCalculator(risk_free_rate=0.02)
            
            # Pipeline
            signals_df = strategy.generate_signals(df)
            result = simulator.execute(signals_df)
            accounting.process_trades(result['trades'], signals_df)
            nav_history = accounting.get_nav_history()
            metrics = calculator.calculate(nav_history, result['trades'])
            
            results[name] = {
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': result['num_trades']
            }
        
        # Assertions
        assert len(results) == 3, "Should have results for all strategies"
        
        # Results should be different (unless by coincidence)
        returns = [r['total_return'] for r in results.values()]
        assert len(set(returns)) > 1, "Strategies should produce different results"
        
        # All metrics should be reasonable
        for name, result in results.items():
            assert result['num_trades'] >= 0, f"Trade count invalid for {name}"
            assert -1 <= result['max_drawdown'] <= 0, f"Max drawdown invalid for {name}"

    def test_data_flow_validation(self, temp_dir):
        """Test data flow integrity between components."""
        data_store = ParquetDataStore(data_dir=temp_dir)
        
        # Generate data with known properties
        original_data = generate_ohlcv_with_trend(
            start_date="2023-01-01",
            end_date="2023-03-31",  # 3 months
            trend=0.001,  # Strong uptrend
            volatility=0.01,
            initial_price=100.0,
            seed=456
        )
        
        symbol = "FLOW_TEST"
        data_store.save(symbol, original_data)
        
        # Step 1: Data loading
        loader = DataLoader(store=data_store)
        loaded_data = loader.download(symbol)
        
        # Validate data integrity
        validate_ohlcv_data(loaded_data)
        assert loaded_data.height == original_data.height
        
        # Step 2: Signal generation
        strategy = SMAStrategy(fast_period=5, slow_period=20)
        signals_data = strategy.generate_signals(loaded_data)
        
        # Validate signals
        assert 'signal' in signals_data.columns
        assert signals_data.height == loaded_data.height
        assert set(signals_data['signal'].unique()) <= {-1, 0, 1}
        
        # Step 3: Trade execution
        simulator = Simulator(initial_cash=50000, commission_bps=10, slippage_pct=0.002)
        execution_result = simulator.execute(signals_data)
        
        # Validate execution
        assert 'trades' in execution_result
        assert 'final_portfolio_value' in execution_result
        assert execution_result['final_portfolio_value'] >= 0
        
        # Step 4: Accounting
        accounting = Accounting(initial_cash=50000)
        accounting.process_trades(execution_result['trades'], signals_data)
        
        # Validate accounting
        nav_history = accounting.get_nav_history()
        assert len(nav_history) > 0
        assert nav_history[0] == 50000  # Initial NAV
        
        # Step 5: Metrics calculation
        calculator = MetricsCalculator()
        metrics = calculator.calculate(nav_history, execution_result['trades'])
        
        # Validate metrics
        required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric] is not None
        
        # Final validation: portfolio value consistency
        final_nav = nav_history[-1]
        final_portfolio = execution_result['final_portfolio_value']
        assert abs(final_nav - final_portfolio) < 0.01, "NAV mismatch"

    def test_error_propagation_and_handling(self, temp_dir):
        """Test error handling and propagation across pipeline."""
        data_store = ParquetDataStore(data_dir=temp_dir)
        
        # Test with corrupted data
        corrupted_data = pl.DataFrame({
            "timestamp": ["2023-01-01", "2023-01-02"],  # Strings instead of datetime
            "open": [100.0, None],  # Null value
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 101.5],
            "volume": [1000, -500]  # Negative volume
        })
        
        # Should handle corrupted data gracefully
        validation = validate_ohlcv_data(corrupted_data)
        assert not validation["valid"]
        assert len(validation["issues"]) > 0
        
        # Test pipeline with invalid strategy configuration
        test_data = generate_ohlcv_with_trend("2023-01-01", "2023-01-31", seed=789)
        data_store.save("VALID_DATA", test_data)
        
        loader = DataLoader(store=data_store)
        valid_data = loader.download("VALID_DATA")
        
        # Test with extreme strategy parameters
        extreme_strategy = SMAStrategy(fast_period=1, slow_period=1000)  # Extreme values
        signals = extreme_strategy.generate_signals(valid_data)
        
        # Should still generate signals (maybe mostly zeros)
        assert 'signal' in signals.columns
        assert signals.height == valid_data.height

    def test_stateless_reproducibility(self, temp_dir):
        """Test that pipeline is stateless and reproducible."""
        data_store = ParquetDataStore(data_dir=temp_dir)
        
        # Generate identical data
        test_data = generate_ohlcv_with_trend(
            start_date="2023-01-01",
            end_date="2023-02-28",
            trend=0.0008,
            volatility=0.025,
            initial_price=150.0,
            seed=999
        )
        
        symbol = "REPRO_TEST"
        data_store.save(symbol, test_data)
        
        def run_pipeline():
            loader = DataLoader(store=data_store)
            data = loader.download(symbol)
            
            strategy = SMAStrategy(fast_period=15, slow_period=45)
            signals = strategy.generate_signals(data)
            
            simulator = Simulator(initial_cash=75000, commission_bps=7, slippage_pct=0.0015)
            result = simulator.execute(signals)
            
            accounting = Accounting(initial_cash=75000)
            accounting.process_trades(result['trades'], signals)
            
            calculator = MetricsCalculator()
            nav_history = accounting.get_nav_history()
            metrics = calculator.calculate(nav_history, result['trades'])
            
            return {
                'num_trades': result['num_trades'],
                'final_value': result['final_portfolio_value'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio']
            }
        
        # Run pipeline multiple times
        result1 = run_pipeline()
        result2 = run_pipeline()
        result3 = run_pipeline()
        
        # Should be identical
        assert result1 == result2 == result3, "Pipeline should be deterministic"

    @pytest.mark.slow
    def test_large_dataset_performance(self, temp_dir, perf_tracker):
        """Test pipeline performance with large dataset."""
        # Generate 5 years of data
        large_data = generate_ohlcv_with_trend(
            start_date="2019-01-01",
            end_date="2023-12-31",
            trend=0.0002,
            volatility=0.018,
            initial_price=80.0,
            seed=555
        )
        
        data_store = ParquetDataStore(data_dir=temp_dir)
        symbol = "LARGE_PERF"
        data_store.save(symbol, large_data)
        
        # Complete pipeline
        perf_tracker.start_timer("full_pipeline")
        
        loader = DataLoader(store=data_store)
        data = loader.download(symbol)
        
        strategy = SMAStrategy(fast_period=20, slow_period=60)
        signals = strategy.generate_signals(data)
        
        simulator = Simulator(initial_cash=200000, commission_bps=3, slippage_pct=0.0008)
        result = simulator.execute(signals)
        
        accounting = Accounting(initial_cash=200000)
        accounting.process_trades(result['trades'], signals)
        
        calculator = MetricsCalculator()
        nav_history = accounting.get_nav_history()
        metrics = calculator.calculate(nav_history, result['trades'])
        
        pipeline_time = perf_tracker.end_timer("full_pipeline")
        
        # Performance assertions
        assert pipeline_time < 3.0, f"Large dataset pipeline too slow: {pipeline_time:.3f}s"
        assert result['num_trades'] > 0, "Should have trades with large dataset"
        assert len(nav_history) == large_data.height, "Should have NAV for all dates"