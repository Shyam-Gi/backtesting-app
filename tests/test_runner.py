"""
Tests for Backtest Runner module
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from backtesting_system.runner import (
    BacktestConfig, run_backtest, run_backtest_with_report, 
    run_multiple_backtests
)
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
import polars as pl


class TestBacktestConfig:
    """Test BacktestConfig class"""
    
    def test_backtest_config_creation(self):
        """Test creating BacktestConfig"""
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={'fast_period': 10, 'slow_period': 50},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_cash=100000,
            commission_bps=5.0,
            slippage_pct=0.001,
            position_size_pct=1.0
        )
        
        assert config.strategy_class == SMAStrategy
        assert config.strategy_params == {'fast_period': 10, 'slow_period': 50}
        assert config.symbol == 'AAPL'
        assert config.start_date == '2020-01-01'
        assert config.end_date == '2020-12-31'
        assert config.initial_cash == 100000
        assert config.commission_bps == 5.0
        assert config.slippage_pct == 0.001
        assert config.position_size_pct == 1.0
    
    def test_backtest_config_defaults(self):
        """Test BacktestConfig with default values"""
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        assert config.initial_cash == 100000
        assert config.commission_bps == 5.0
        assert config.slippage_pct == 0.001
        assert config.position_size_pct == 1.0
    
    def test_to_dict(self):
        """Test converting BacktestConfig to dictionary"""
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={'fast_period': 10},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['strategy_name'] == 'SMAStrategy'
        assert config_dict['strategy_params'] == {'fast_period': 10}
        assert config_dict['symbol'] == 'AAPL'
        assert config_dict['start_date'] == '2020-01-01'
        assert config_dict['end_date'] == '2020-12-31'
        assert config_dict['initial_cash'] == 100000


class TestRunBacktest:
    """Test run_backtest function"""
    
    @patch('backtesting_system.runner.DataLoader')
    @patch('backtesting_system.runner.Simulator')
    @patch('backtesting_system.runner.Accounting')
    @patch('backtesting_system.runner.MetricsCalculator')
    def test_run_backtest_success(self, mock_calculator, mock_accounting, 
                                  mock_simulator, mock_loader):
        """Test successful backtest execution"""
        # Mock data
        mock_data = pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02'],
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000000, 1100000]
        })
        mock_loader.return_value.load.return_value = mock_data
        
        # Mock strategy signals
        mock_signals = mock_data.with_columns([
            pl.lit(1).alias('signal')
        ])
        
        # Mock simulator result
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.execute.return_value = {
            'trades': [
                {
                    'timestamp': '2020-01-01',
                    'side': 'BUY',
                    'price': 101.0,
                    'quantity': 100,
                    'commission': 5.05,
                    'slippage': 0.101
                }
            ],
            'final_portfolio_value': 95000.0,
            'total_return': -0.05,
            'num_trades': 1
        }
        mock_simulator.return_value = mock_simulator_instance
        
        # Mock accounting
        mock_accounting_instance = MagicMock()
        mock_accounting_instance.get_nav_history.return_value = [
            {'timestamp': '2020-01-01', 'nav': 100000.0},
            {'timestamp': '2020-01-02', 'nav': 95000.0}
        ]
        mock_accounting_instance.get_positions.return_value = {}
        mock_accounting_instance.get_cash_history.return_value = [
            {'timestamp': '2020-01-01', 'cash': 0.0},
            {'timestamp': '2020-01-02', 'cash': 10000.0}
        ]
        mock_accounting.return_value = mock_accounting_instance
        
        # Mock metrics
        mock_calculator_instance = MagicMock()
        mock_calculator_instance.calculate.return_value = {
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.05,
            'win_rate': 0.6
        }
        mock_calculator.return_value = mock_calculator_instance
        
        # Create config
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={'fast_period': 10, 'slow_period': 50},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        # Run backtest
        result = run_backtest(config)
        
        # Verify result
        assert result['success'] is True
        assert result['final_portfolio_value'] == 95000.0
        assert result['total_return'] == -0.05
        assert result['num_trades'] == 1
        assert result['config']['strategy_name'] == 'SMAStrategy'
        assert result['metrics']['sharpe_ratio'] == 1.5
        assert len(result['nav_history']) == 2
        assert result['execution_time_seconds'] > 0
        assert result['data_points'] == 2
    
    @patch('backtesting_system.runner.DataLoader')
    def test_run_backtest_data_loading_error(self, mock_loader):
        """Test backtest with data loading error"""
        mock_loader.side_effect = Exception("Data loading failed")
        
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        result = run_backtest(config)
        
        assert result['success'] is False
        assert 'Data loading failed' in result['error']
        assert result['final_portfolio_value'] == 100000  # Initial cash
        assert result['num_trades'] == 0
        assert result['data_points'] == 0
    
    def test_run_backtest_invalid_strategy_params(self):
        """Test backtest with invalid strategy parameters"""
        # Use a strategy that requires specific parameters
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={'invalid_param': 10},  # Missing required params
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        # This should handle the invalid params gracefully
        result = run_backtest(config)
        
        # The exact behavior depends on strategy implementation
        # but it should not crash and should have success=False
        assert isinstance(result, dict)
        assert 'success' in result


class TestRunBacktestWithReport:
    """Test run_backtest_with_report function"""
    
    @patch('backtesting_system.runner.run_backtest')
    @patch('backtesting_system.runner.BacktestReport')
    @patch('backtesting_system.runner.Accounting')
    @patch('backtesting_system.runner.DataLoader')
    def test_run_backtest_with_report_success(self, mock_loader, mock_accounting, 
                                              mock_report, mock_run_backtest):
        """Test successful backtest with report generation"""
        # Mock successful backtest
        mock_run_backtest.return_value = {
            'success': True,
            'trades': [{'timestamp': '2020-01-01', 'side': 'BUY', 'price': 100.0}],
            'final_portfolio_value': 105000.0,
            'total_return': 0.05,
            'num_trades': 1,
            'nav_history': [{'timestamp': '2020-01-01', 'nav': 105000.0}],
            'positions': {},
            'cash_history': [{'timestamp': '2020-01-01', 'cash': 5000.0}],
            'metrics': {'sharpe_ratio': 1.5},
            'config': {'symbol': 'AAPL', 'start_date': '2020-01-01', 'end_date': '2020-01-31'}
        }
        
        # Mock data loading for report
        mock_data = pl.DataFrame({
            'timestamp': ['2020-01-01'],
            'close': [100.0]
        })
        mock_loader.return_value.load.return_value = mock_data
        
        # Mock report generation
        mock_report_instance = MagicMock()
        mock_report_instance.generate_full_report.return_value = {
            'json': 'report.json',
            'csv': 'trades.csv',
            'html': 'chart.html'
        }
        mock_report.return_value = mock_report_instance
        
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        result = run_backtest_with_report(config, "test_output")
        
        assert result['success'] is True
        assert 'report_files' in result
        assert result['report_files']['json'] == 'report.json'
        assert result['report_files']['csv'] == 'trades.csv'
        assert result['report_files']['html'] == 'chart.html'
    
    @patch('backtesting_system.runner.run_backtest')
    def test_run_backtest_with_report_failure(self, mock_run_backtest):
        """Test backtest with report when backtest fails"""
        mock_run_backtest.return_value = {
            'success': False,
            'error': 'Backtest failed'
        }
        
        config = BacktestConfig(
            strategy_class=SMAStrategy,
            strategy_params={},
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-01-31'
        )
        
        result = run_backtest_with_report(config)
        
        assert result['success'] is False
        assert result['error'] == 'Backtest failed'
        assert result['report_files'] == {}


class TestRunMultipleBacktests:
    """Test run_multiple_backtests function"""
    
    @patch('backtesting_system.runner.run_backtest')
    def test_run_multiple_backtests_success(self, mock_run_backtest):
        """Test running multiple backtests successfully"""
        # Mock successful backtest results
        def run_backtest_side_effect(config):
            if config.symbol == 'AAPL':
                return {
                    'success': True,
                    'final_portfolio_value': 105000.0,
                    'total_return': 0.05,
                    'num_trades': 1,
                    'execution_time_seconds': 0.5
                }
            elif config.symbol == 'MSFT':
                return {
                    'success': True,
                    'final_portfolio_value': 95000.0,
                    'total_return': -0.05,
                    'num_trades': 2,
                    'execution_time_seconds': 0.6
                }
            else:
                return {'success': False, 'error': 'Unknown symbol'}
        
        mock_run_backtest.side_effect = run_backtest_side_effect
        
        configs = [
            BacktestConfig(
                strategy_class=SMAStrategy,
                strategy_params={},
                symbol='AAPL',
                start_date='2020-01-01',
                end_date='2020-01-31'
            ),
            BacktestConfig(
                strategy_class=MomentumStrategy,
                strategy_params={},
                symbol='MSFT',
                start_date='2020-01-01',
                end_date='2020-01-31'
            )
        ]
        
        results = run_multiple_backtests(configs)
        
        assert len(results) == 2
        # Check that we got both results, order may vary due to parallel execution
        portfolio_values = [r['final_portfolio_value'] for r in results]
        assert 105000.0 in portfolio_values
        assert 95000.0 in portfolio_values
        assert all(r['success'] for r in results)
        assert mock_run_backtest.call_count == 2
    
    @patch('backtesting_system.runner.run_backtest')
    def test_run_multiple_backtests_mixed_results(self, mock_run_backtest):
        """Test running multiple backtests with mixed success/failure"""
        def run_backtest_side_effect(config):
            if config.symbol == 'AAPL':
                return {
                    'success': True,
                    'final_portfolio_value': 105000.0,
                    'total_return': 0.05,
                    'num_trades': 1,
                    'execution_time_seconds': 0.5
                }
            elif config.symbol == 'INVALID':
                return {
                    'success': False,
                    'error': 'Data loading failed',
                    'final_portfolio_value': 100000.0,
                    'total_return': 0.0,
                    'num_trades': 0,
                    'execution_time_seconds': 0.1
                }
            else:
                return {'success': False, 'error': 'Unknown symbol'}
        
        mock_run_backtest.side_effect = run_backtest_side_effect
        
        configs = [
            BacktestConfig(
                strategy_class=SMAStrategy,
                strategy_params={},
                symbol='AAPL',
                start_date='2020-01-01',
                end_date='2020-01-31'
            ),
            BacktestConfig(
                strategy_class=MomentumStrategy,
                strategy_params={},
                symbol='INVALID',
                start_date='2020-01-01',
                end_date='2020-01-31'
            )
        ]
        
        results = run_multiple_backtests(configs)
        
        assert len(results) == 2
        # Check that we have one success and one failure
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        assert len(successes) == 1
        assert len(failures) == 1
        assert successes[0]['final_portfolio_value'] == 105000.0
        assert failures[0]['error'] == 'Data loading failed'
    
    def test_run_multiple_backtests_empty_list(self):
        """Test running multiple backtests with empty config list"""
        results = run_multiple_backtests([])
        
        assert len(results) == 0
        assert isinstance(results, list)


class TestIntegration:
    """Integration tests for runner module"""
    
    def test_end_to_end_simple_backtest(self):
        """Test end-to-end backtest with minimal setup"""
        # This test requires actual data loading and strategy execution
        # It should be used carefully as it makes network calls
        
        # Create minimal test data
        test_data = pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Mock the data loader to return our test data
        with patch('backtesting_system.runner.DataLoader') as mock_loader:
            mock_loader.return_value.load.return_value = test_data
            
            config = BacktestConfig(
                strategy_class=SMAStrategy,
                strategy_params={'fast_period': 2, 'slow_period': 3},
                symbol='TEST',
                start_date='2020-01-01',
                end_date='2020-01-05',
                initial_cash=10000,
                commission_bps=0,  # No commission for simple test
                slippage_pct=0,   # No slippage for simple test
            )
            
            result = run_backtest(config)
            
            # Verify basic structure
            assert 'success' in result
            assert 'final_portfolio_value' in result
            assert 'total_return' in result
            assert 'num_trades' in result
            assert 'metrics' in result
            assert 'execution_time_seconds' in result
            
            # Should not crash and should return some result
            assert isinstance(result['success'], bool)
            assert isinstance(result['final_portfolio_value'], (int, float))
            assert isinstance(result['total_return'], (int, float))
            assert isinstance(result['num_trades'], int)
            assert isinstance(result['execution_time_seconds'], (int, float))