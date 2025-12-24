"""Tests for reporting module.

Unit tests for report generation, JSON exports, CSV exports, and chart creation.
"""

import pytest
import polars as pl
import json
import csv
from pathlib import Path
import tempfile
from datetime import datetime

from backtesting_system.reporting import BacktestReport
from backtesting_system.accounting import Accounting
from backtesting_system.metrics import PerformanceMetrics
from backtesting_system.simulator import Trade


class TestBacktestReport:
    """Test suite for BacktestReport class."""
    
    @pytest.fixture
    def sample_accounting(self):
        """Create sample accounting data."""
        acct = Accounting(initial_cash=100000)
        
        # Create sample nav history
        dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        values = [100000, 101000, 102000, 101500, 103000]
        
        nav_history = pl.DataFrame({
            'timestamp': dates,
            'cash': [50000, 49500, 49000, 50750, 51500],
            'position_value': [50000, 51500, 53000, 50750, 51500],
            'total_value': values,
            'position_quantity': [500, 515, 530, 507, 515],
            'unrealized_pnl': [0, 1500, 3000, 750, 1500],
            'realized_pnl': [0, 0, 0, 0, 0],
            'total_pnl': [0, 1500, 3000, 750, 1500]
        })
        
        acct.portfolio_history = []
        # Manually add portfolio snapshots
        for i, date in enumerate(dates):
            from backtesting_system.accounting import PortfolioSnapshot
            snapshot = PortfolioSnapshot(
                timestamp=date,
                cash=float(nav_history['cash'][i]),
                position_value=float(nav_history['position_value'][i]),
                total_value=float(nav_history['total_value'][i]),
                position_quantity=int(nav_history['position_quantity'][i]),
                unrealized_pnl=float(nav_history['unrealized_pnl'][i]),
                realized_pnl=float(nav_history['realized_pnl'][i]),
                total_pnl=float(nav_history['total_pnl'][i])
            )
            acct.portfolio_history.append(snapshot)
        
        return acct
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return PerformanceMetrics(
            total_return=0.03,
            cagr=0.025,
            annualized_return=0.026,
            daily_return_mean=0.0001,
            daily_return_std=0.01,
            volatility=0.16,
            max_drawdown=0.02,
            max_drawdown_duration=5,
            calmar_ratio=1.25,
            sharpe_ratio=0.8,
            sortino_ratio=1.2,
            information_ratio=0.3,
            beta=0.9,
            alpha=0.01,
            win_rate=0.6,
            avg_win=500.0,
            avg_loss=-300.0,
            profit_factor=1.8,
            expectancy=50.0,
            trade_count=10,
            avg_holding_period=15.5,
            turnover=0.5,
            var_95=-0.02,
            skewness=0.1,
            kurtosis=0.5
        )
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades."""
        return [
            Trade('2020-01-02', 'BUY', 100.0, 100, 5.0, 1.0, 505.0),
            Trade('2020-01-04', 'SELL', 102.0, 100, 5.1, 1.02, 5.1),
            Trade('2020-01-05', 'BUY', 103.0, 50, 2.575, 0.515, 52.575)
        ]
    
    def test_initialization(self):
        """Test report generator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            assert report.output_dir == Path(temp_dir)
            assert report.output_dir.exists()
    
    def test_generate_full_report(self, sample_accounting, sample_metrics, sample_trades):
        """Test full report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            files = report.generate_full_report(
                accounting=sample_accounting,
                metrics=sample_metrics,
                trades=sample_trades,
                strategy_name='SMA_Crossover',
                symbol='AAPL',
                start_date='2020-01-01',
                end_date='2020-01-05',
                initial_cash=100000,
                include_charts=False  # Disable charts for testing
            )
            
            # Should generate required files
            assert 'json_summary' in files
            assert 'csv_trades' in files
            assert 'csv_portfolio' in files
            
            # Check files exist
            for file_path in files.values():
                assert Path(file_path).exists()
    
    def test_json_summary_generation(self, sample_accounting, sample_metrics, sample_trades):
        """Test JSON summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            json_path = report._generate_json_summary(
                accounting=sample_accounting,
                metrics=sample_metrics,
                trades=sample_trades,
                strategy_name='Test_Strategy',
                symbol='TEST',
                start_date='2020-01-01',
                end_date='2020-01-05',
                initial_cash=100000,
                base_filename='test_report'
            )
            
            # Check file exists
            assert json_path.exists()
            
            # Load and verify JSON structure
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check required sections
            assert 'metadata' in data
            assert 'performance_metrics' in data
            assert 'portfolio_breakdown' in data
            assert 'trade_summary' in data
            
            # Check metadata
            metadata = data['metadata']
            assert metadata['strategy_name'] == 'Test_Strategy'
            assert metadata['symbol'] == 'TEST'
            assert metadata['initial_cash'] == 100000
            assert metadata['trade_count'] == len(sample_trades)
            
            # Check performance metrics
            perf = data['performance_metrics']
            assert 'returns' in perf
            assert 'risk_metrics' in perf
            assert 'risk_adjusted_returns' in perf
            assert 'trade_statistics' in perf
            assert 'efficiency_metrics' in perf
            
            # Check specific values
            returns = perf['returns']
            assert returns['total_return'] == 0.03
            assert perf['risk_adjusted_returns']['sharpe_ratio'] == 0.8
            
            # Check portfolio breakdown
            portfolio = data['portfolio_breakdown']
            assert portfolio['initial_cash'] == 100000
            assert portfolio['realized_pnl'] == 0.0
            
            # Check trade summary
            trade_summary = data['trade_summary']
            assert trade_summary['total_trades'] == len(sample_trades)
            assert trade_summary['buy_trades'] == 2  # First and third trades
            assert trade_summary['sell_trades'] == 1  # Second trade
    
    def test_csv_trades_generation(self, sample_trades):
        """Test CSV trade log generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            csv_path = report._generate_csv_trades(sample_trades, 'test_trades')
            
            # Check file exists
            assert csv_path.exists()
            
            # Read CSV and verify structure
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            # Check headers
            expected_headers = ['timestamp', 'side', 'price', 'quantity', 'commission', 'slippage', 'total_cost']
            assert headers == expected_headers
            
            # Check data rows
            assert len(rows) == len(sample_trades)
            
            # Check first trade
            first_row = rows[0]
            assert first_row[0] == '2020-01-02'  # timestamp
            assert first_row[1] == 'BUY'        # side
            assert first_row[2] == '100.0'       # price
            assert first_row[3] == '100'          # quantity
    
    def test_csv_portfolio_generation(self, sample_accounting):
        """Test CSV portfolio history generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            csv_path = report._generate_csv_portfolio_history(sample_accounting, 'test_portfolio')
            
            # Check file exists
            assert csv_path.exists()
            
            # Read CSV and verify structure
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            # Check headers
            expected_headers = ['timestamp', 'cash', 'position_value', 'total_value', 'position_quantity', 'unrealized_pnl', 'realized_pnl', 'total_pnl']
            assert headers == expected_headers
            
            # Check data rows count
            assert len(rows) == len(sample_accounting.portfolio_history)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            # Test empty trades
            csv_path = report._generate_csv_trades([], 'empty_trades')
            assert csv_path.exists()
            
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            assert len(rows) == 0  # No data rows
            
            # Test empty accounting
            empty_acct = Accounting(initial_cash=100000)
            csv_path = report._generate_csv_portfolio_history(empty_acct, 'empty_portfolio')
            assert csv_path.exists()
    
    def test_trade_summary_calculation(self, sample_trades):
        """Test trade summary statistics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            summary = report._summarize_trades(sample_trades)
            
            assert summary['total_trades'] == len(sample_trades)
            assert summary['buy_trades'] == 2
            assert summary['sell_trades'] == 1
            assert abs(summary['total_commission'] - sum(trade.commission for trade in sample_trades)) < 0.01
            assert abs(summary['total_slippage'] - sum(trade.slippage for trade in sample_trades)) < 0.01
            expected_avg = sum(trade.quantity for trade in sample_trades) / len(sample_trades)
        assert abs(summary['avg_trade_size'] - expected_avg) < 1.0  # Allow some rounding difference
    
    def test_filename_generation(self, sample_accounting, sample_metrics, sample_trades):
        """Test that generated files have consistent naming."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            files = report.generate_full_report(
                accounting=sample_accounting,
                metrics=sample_metrics,
                trades=sample_trades,
                strategy_name='TestStrategy',
                symbol='AAPL',
                start_date='2020-01-01',
                end_date='2020-01-05',
                initial_cash=100000,
                include_charts=False
            )
            
            # All files should have the same base filename pattern
            base_patterns = []
            for file_type, file_path in files.items():
                path_obj = Path(file_path)
                filename = path_obj.name
                # Extract base name (remove suffix)
                base_name = None
                if 'json_summary' in file_type:
                    base_name = filename.replace('_summary.json', '')
                elif 'csv_trades' in file_type:
                    base_name = filename.replace('_trades.csv', '')
                elif 'csv_portfolio' in file_type:
                    base_name = filename.replace('_portfolio.csv', '')
                
                if base_name is not None:
                    base_patterns.append(base_name)
            
            # All base patterns should be identical
            if base_patterns:
                assert all(pattern == base_patterns[0] for pattern in base_patterns)
                
                # Base pattern should contain strategy and symbol
                base_pattern = base_patterns[0]
                assert 'TestStrategy' in base_pattern
                assert 'AAPL' in base_pattern


class TestReportGenerationEdgeCases:
    """Test edge cases for report generation."""
    
    def test_large_number_of_trades(self):
        """Test handling large number of trades."""
        # Create many trades
        trades = []
        for i in range(1000):
            trades.append(Trade(
                f'2020-01-{i%28+1:02d}', 
                'BUY' if i % 2 == 0 else 'SELL',
                100.0 + i * 0.1,
                100,
                5.0,
                1.0,
                505.0
            ))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            # Should handle large trade list
            csv_path = report._generate_csv_trades(trades, 'large_trades')
            assert csv_path.exists()
            
            # Verify row count
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            assert len(rows) == 1000
    
    def test_zero_values_handling(self):
        """Test handling of zero values in metrics."""
        zero_metrics = PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_return=0.0,
            daily_return_mean=0.0,
            daily_return_std=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            information_ratio=0.0,
            beta=0.0,
            alpha=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            trade_count=0,
            avg_holding_period=0.0,
            turnover=0.0,
            var_95=0.0,
            skewness=0.0,
            kurtosis=0.0
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report = BacktestReport(output_dir=temp_dir)
            
            # Should handle zero metrics gracefully
            json_path = report._generate_json_summary(
                accounting=Accounting(initial_cash=100000),
                metrics=zero_metrics,
                trades=[],
                strategy_name='Zero_Test',
                symbol='ZERO',
                start_date='2020-01-01',
                end_date='2020-01-01',
                initial_cash=100000,
                base_filename='zero_test'
            )
            
            assert json_path.exists()
            
            # Load and verify JSON contains zero values
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            perf = data['performance_metrics']
            assert perf['returns']['total_return'] == 0.0
            assert perf['risk_metrics']['volatility'] == 0.0
            assert perf['trade_statistics']['trade_count'] == 0