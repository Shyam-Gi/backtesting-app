"""Performance regression tests for backtesting system."""

import pytest
import time
import polars as pl
import numpy as np
from pathlib import Path
import tempfile

from backtesting_system.data_loader import DataLoader, ParquetDataStore
from backtesting_system.simulator import Simulator
from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator
from strategies.sma_strategy import SMAStrategy

from tests.utils import generate_ohlcv_with_trend


class TestPerformanceBenchmarks:
    """Performance benchmarks and regression tests."""

    # Performance targets (in seconds)
    TARGETS = {
        "data_load_1year": 0.1,
        "data_load_5years": 0.3,
        "signal_generation_1year": 0.05,
        "signal_generation_5years": 0.1,
        "trade_execution_1year": 0.1,
        "trade_execution_5years": 0.2,
        "accounting_1year": 0.05,
        "accounting_5years": 0.1,
        "metrics_1year": 0.05,
        "metrics_5years": 0.1,
        "full_pipeline_1year": 0.5,
        "full_pipeline_5years": 1.0
    }

    @pytest.fixture
    def benchmark_data(self):
        """Generate benchmark datasets of different sizes."""
        return {
            "1year": generate_ohlcv_with_trend(
                start_date="2023-01-01", 
                end_date="2023-12-31",
                trend=0.0005,
                volatility=0.02,
                seed=1001
            ),
            "5years": generate_ohlcv_with_trend(
                start_date="2019-01-01", 
                end_date="2023-12-31",
                trend=0.0003,
                volatility=0.018,
                seed=1002
            )
        }

    def test_data_loading_performance(self, benchmark_data, temp_dir, perf_tracker):
        """Benchmark data loading performance."""
        data_store = ParquetDataStore(data_dir=temp_dir)
        
        for size, data in benchmark_data.items():
            # Save data first
            symbol = f"BENCH_{size.upper()}"
            data_store.save(symbol, data)
            
            # Benchmark loading
            perf_tracker.start_timer(f"data_load_{size}")
            loader = DataLoader(store=data_store)
            loaded_data = loader.download(symbol)
            load_time = perf_tracker.end_timer(f"data_load_{size}")
            
            target = self.TARGETS[f"data_load_{size}"]
            
            assert load_time <= target, \
                f"Data loading ({size}) too slow: {load_time:.3f}s > {target:.3f}s"
            
            # Verify data integrity
            assert loaded_data.height == data.height
            assert list(loaded_data.columns) == list(data.columns)

    def test_signal_generation_performance(self, benchmark_data, perf_tracker):
        """Benchmark strategy signal generation."""
        strategy = SMAStrategy(fast_period=10, slow_period=50)
        
        for size, data in benchmark_data.items():
            perf_tracker.start_timer(f"signal_generation_{size}")
            signals = strategy.generate_signals(data)
            signal_time = perf_tracker.end_timer(f"signal_generation_{size}")
            
            target = self.TARGETS[f"signal_generation_{size}"]
            
            assert signal_time <= target, \
                f"Signal generation ({size}) too slow: {signal_time:.3f}s > {target:.3f}s"
            
            # Verify signals
            assert 'signal' in signals.columns
            assert signals.height == data.height

    def test_trade_execution_performance(self, benchmark_data, perf_tracker):
        """Benchmark trade execution performance."""
        for size, data in benchmark_data.items():
            # Generate signals first
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            signals = strategy.generate_signals(data)
            
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            
            perf_tracker.start_timer(f"trade_execution_{size}")
            result = simulator.execute(signals)
            execution_time = perf_tracker.end_timer(f"trade_execution_{size}")
            
            target = self.TARGETS[f"trade_execution_{size}"]
            
            assert execution_time <= target, \
                f"Trade execution ({size}) too slow: {execution_time:.3f}s > {target:.3f}s"
            
            # Verify execution results
            assert 'trades' in result
            assert 'final_portfolio_value' in result

    def test_accounting_performance(self, benchmark_data, perf_tracker):
        """Benchmark portfolio accounting performance."""
        for size, data in benchmark_data.items():
            # Setup: generate signals and trades
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            signals = strategy.generate_signals(data)
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            result = simulator.execute(signals)
            
            accounting = Accounting(initial_cash=100000)
            
            perf_tracker.start_timer(f"accounting_{size}")
            accounting.process_trades(result['trades'], signals)
            nav_history = accounting.get_nav_history()
            accounting_time = perf_tracker.end_timer(f"accounting_{size}")
            
            target = self.TARGETS[f"accounting_{size}"]
            
            assert accounting_time <= target, \
                f"Accounting ({size}) too slow: {accounting_time:.3f}s > {target:.3f}s"
            
            # Verify accounting results
            assert len(nav_history) == data.height
            assert nav_history[0] == 100000

    def test_metrics_calculation_performance(self, benchmark_data, perf_tracker):
        """Benchmark performance metrics calculation."""
        for size, data in benchmark_data.items():
            # Setup: get accounting data
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            signals = strategy.generate_signals(data)
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            result = simulator.execute(signals)
            
            accounting = Accounting(initial_cash=100000)
            accounting.process_trades(result['trades'], signals)
            nav_history = accounting.get_nav_history()
            
            calculator = MetricsCalculator(risk_free_rate=0.02)
            
            perf_tracker.start_timer(f"metrics_{size}")
            metrics = calculator.calculate(nav_history, result['trades'])
            metrics_time = perf_tracker.end_timer(f"metrics_{size}")
            
            target = self.TARGETS[f"metrics_{size}"]
            
            assert metrics_time <= target, \
                f"Metrics calculation ({size}) too slow: {metrics_time:.3f}s > {target:.3f}s"
            
            # Verify metrics
            required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
            for metric in required_metrics:
                assert metric in metrics

    def test_full_pipeline_performance(self, benchmark_data, temp_dir, perf_tracker):
        """Benchmark complete backtest pipeline performance."""
        data_store = ParquetDataStore(data_dir=temp_dir)
        
        for size, data in benchmark_data.items():
            symbol = f"PIPELINE_{size.upper()}"
            data_store.save(symbol, data)
            
            perf_tracker.start_timer(f"full_pipeline_{size}")
            
            # Complete pipeline
            loader = DataLoader(store=data_store)
            df = loader.download(symbol)
            
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            signals = strategy.generate_signals(df)
            
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            result = simulator.execute(signals)
            
            accounting = Accounting(initial_cash=100000)
            accounting.process_trades(result['trades'], signals)
            
            calculator = MetricsCalculator(risk_free_rate=0.02)
            nav_history = accounting.get_nav_history()
            metrics = calculator.calculate(nav_history, result['trades'])
            
            pipeline_time = perf_tracker.end_timer(f"full_pipeline_{size}")
            
            target = self.TARGETS[f"full_pipeline_{size}"]
            
            assert pipeline_time <= target, \
                f"Full pipeline ({size}) too slow: {pipeline_time:.3f}s > {target:.3f}s"
            
            # Verify results
            assert result['num_trades'] > 0
            assert metrics['total_return'] is not None

    def test_memory_usage_efficiency(self, benchmark_data, perf_tracker):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test with 5-year data
        data = benchmark_data["5years"]
        
        # Data loading
        data_store = ParquetDataStore(data_dir=tempfile.mkdtemp())
        data_store.save("MEMORY_TEST", data)
        loader = DataLoader(store=data_store)
        df = loader.download("MEMORY_TEST")
        
        # Signal generation
        strategy = SMAStrategy(fast_period=10, slow_period=50)
        signals = strategy.generate_signals(df)
        
        # Trade execution
        simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
        result = simulator.execute(signals)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory should be reasonable (less than 500MB for 5-year dataset)
        assert memory_increase < 500, \
            f"Memory usage too high: {memory_increase:.1f}MB for 5-year dataset"

    def test_scalability_analysis(self, perf_tracker):
        """Analyze performance scaling with dataset size."""
        sizes = [
            ("1month", "2023-01-01", "2023-01-31"),
            ("3months", "2023-01-01", "2023-03-31"),
            ("6months", "2023-01-01", "2023-06-30"),
            ("1year", "2023-01-01", "2023-12-31"),
            ("2years", "2022-01-01", "2023-12-31")
        ]
        
        results = {}
        
        for name, start_date, end_date in sizes:
            # Generate data
            data = generate_ohlcv_with_trend(
                start_date=start_date,
                end_date=end_date,
                seed=2000 + len(name)
            )
            
            # Benchmark signal generation (core computation)
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            
            perf_tracker.start_timer(f"scalability_{name}")
            signals = strategy.generate_signals(data)
            time_taken = perf_tracker.end_timer(f"scalability_{name}")
            
            results[name] = {
                'rows': data.height,
                'time': time_taken,
                'time_per_row': time_taken / data.height * 1000000  # microseconds per row
            }
        
        # Check for linear scaling (time per row should be roughly constant)
        time_per_rows = [r['time_per_row'] for r in results.values()]
        avg_time_per_row = np.mean(time_per_rows)
        std_time_per_row = np.std(time_per_rows)
        
        # Coefficient of variation should be low (consistent performance)
        cv = std_time_per_row / avg_time_per_row
        assert cv < 0.5, \
            f"Poor scaling consistency, CV={cv:.2f}, times per row: {time_per_rows}"

    @pytest.mark.slow
    def test_long_term_performance_regression(self, benchmark_data):
        """Test for performance regression over time."""
        # This test records current performance for future comparison
        data_store = ParquetDataStore(data_dir=tempfile.mkdtemp())
        
        # Test 1-year dataset (most common use case)
        symbol = "REGRESSION_1YEAR"
        data_store.save(symbol, benchmark_data["1year"])
        
        # Run complete pipeline
        start_time = time.time()
        
        loader = DataLoader(store=data_store)
        df = loader.download(symbol)
        
        strategy = SMAStrategy(fast_period=10, slow_period=50)
        signals = strategy.generate_signals(df)
        
        simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
        result = simulator.execute(signals)
        
        accounting = Accounting(initial_cash=100000)
        accounting.process_trades(result['trades'], signals)
        
        calculator = MetricsCalculator(risk_free_rate=0.02)
        nav_history = accounting.get_nav_history()
        metrics = calculator.calculate(nav_history, result['trades'])
        
        total_time = time.time() - start_time
        
        # Performance regression check (should be under target)
        assert total_time <= self.TARGETS["full_pipeline_1year"], \
            f"Performance regression detected: {total_time:.3f}s > {self.TARGETS['full_pipeline_1year']:.3f}s"
        
        # Record performance for monitoring
        print(f"\nPerformance Baseline (1-year dataset):")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Data points: {benchmark_data['1year'].height}")
        print(f"  Time per point: {total_time / benchmark_data['1year'].height * 1000:.3f}ms")
        print(f"  Trades executed: {result['num_trades']}")
        print(f"  Total return: {metrics['total_return']:.2%}")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}")

    def test_concurrent_performance(self, benchmark_data, perf_tracker):
        """Test performance under concurrent load."""
        import threading
        import concurrent.futures
        
        def run_single_backtest(test_id):
            """Run a single backtest for concurrent testing."""
            # Use copy of data to avoid conflicts
            data = benchmark_data["1year"].clone()
            
            # Complete pipeline
            strategy = SMAStrategy(fast_period=10, slow_period=50)
            signals = strategy.generate_signals(data)
            simulator = Simulator(initial_cash=100000, commission_bps=5, slippage_pct=0.001)
            result = simulator.execute(signals)
            
            return {
                'test_id': test_id,
                'num_trades': result['num_trades'],
                'final_value': result['final_portfolio_value']
            }
        
        # Test with 4 concurrent backtests (simulating multi-core usage)
        perf_tracker.start_timer("concurrent_4_backtests")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_single_backtest, i) for i in range(4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = perf_tracker.end_timer("concurrent_4_backtests")
        
        # Should be faster than running sequentially
        perf_tracker.start_timer("sequential_4_backtests")
        sequential_results = [run_single_backtest(i) for i in range(4)]
        sequential_time = perf_tracker.end_timer("sequential_4_backtests")
        
        # Concurrent should be faster (but allow some overhead)
        speedup = sequential_time / concurrent_time
        assert speedup > 2.0, \
            f"Insufficient parallelization: speedup={speedup:.2f}x (want >2.0x)"
        
        # Results should be consistent
        for i, result in enumerate(results):
            sequential_result = sequential_results[i]
            assert result['num_trades'] == sequential_result['num_trades']
            assert abs(result['final_value'] - sequential_result['final_value']) < 0.01