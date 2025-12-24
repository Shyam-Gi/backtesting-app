"""
Tests for Bias Validation module
"""

import pytest
import polars as pl
from backtesting_system.validation import (
    ValidationResult, BiasValidator
)


class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult"""
        result = ValidationResult(
            name="Test Validation",
            passed=True,
            message="All good",
            severity="info"
        )
        
        assert result.name == "Test Validation"
        assert result.passed is True
        assert result.message == "All good"
        assert result.severity == "info"
    
    def test_validation_result_str(self):
        """Test ValidationResult string representation"""
        result_passed = ValidationResult(
            name="Test",
            passed=True,
            message="OK",
            severity="warning"
        )
        result_failed = ValidationResult(
            name="Test",
            passed=False,
            message="Failed",
            severity="error"
        )
        
        assert "✅ Test: OK" in str(result_passed)
        assert "❌ Test: Failed" in str(result_failed)
        assert "🚨" in str(result_failed)


class TestBiasValidator:
    """Test BiasValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = BiasValidator()
        
        # Sample valid OHLCV data
        self.valid_data = pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        # Sample signals data
        self.signals = pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'signal': [1, 0, -1]
        })
        
        # Sample trades
        self.trades = [
            {'timestamp': '2020-01-01', 'side': 'BUY', 'price': 101.0, 'quantity': 100},
            {'timestamp': '2020-01-03', 'side': 'SELL', 'price': 103.0, 'quantity': 100}
        ]
    
    def test_validator_initialization(self):
        """Test BiasValidator initialization"""
        validator = BiasValidator(warn_on_low_costs=False)
        assert validator.warn_on_low_costs is False
        assert validator.results == []
    
    def test_validate_data_quality_success(self):
        """Test successful data quality validation"""
        result = self.validator.validate_data_quality(self.valid_data)
        
        assert result.name == "Data Quality"
        assert result.passed is True
        assert "good" in result.message.lower()
        assert result.severity == "info"
    
    def test_validate_data_quality_missing_columns(self):
        """Test data quality validation with missing columns"""
        bad_data = pl.DataFrame({
            'timestamp': ['2020-01-01'],
            'open': [100.0]
            # Missing other required columns
        })
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "Missing columns" in result.message
    
    def test_validate_data_quality_null_values(self):
        """Test data quality validation with null values"""
        bad_data = self.valid_data.with_columns([
            pl.lit(None).alias('close')
        ])
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "null values" in result.message.lower()
    
    def test_validate_data_quality_invalid_ohlc(self):
        """Test data quality validation with invalid OHLC relationships"""
        bad_data = self.valid_data.with_columns([
            pl.lit(95.0).alias('high')  # High < Low
        ])
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "High < Low" in result.message
    
    def test_validate_data_quality_negative_prices(self):
        """Test data quality validation with negative prices"""
        bad_data = self.valid_data.with_columns([
            pl.lit(-100.0).alias('close')
        ])
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "non-positive prices" in result.message
    
    def test_validate_data_quality_negative_volume(self):
        """Test data quality validation with negative volume"""
        bad_data = self.valid_data.with_columns([
            pl.lit(-1000).alias('volume')
        ])
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "negative volume" in result.message
    
    def test_validate_data_quality_duplicates(self):
        """Test data quality validation with duplicate timestamps"""
        bad_data = pl.concat([
            self.valid_data,
            self.valid_data.slice(0, 1)  # Duplicate first row
        ])
        
        result = self.validator.validate_data_quality(bad_data)
        
        assert result.passed is False
        assert result.severity == "warning"
        assert "duplicate timestamps" in result.message
    
    def test_validate_lookahead_bias_success(self):
        """Test successful lookahead bias validation"""
        result = self.validator.validate_lookahead_bias(self.valid_data, self.signals)
        
        assert result.name == "Look-ahead Bias"
        assert result.passed is True
        assert result.severity == "info"
    
    def test_validate_lookahead_bias_no_timestamp(self):
        """Test lookahead bias validation with missing timestamp"""
        data_no_ts = self.valid_data.drop('timestamp')
        
        result = self.validator.validate_lookahead_bias(data_no_ts, self.signals)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "timestamp" in result.message.lower()
    
    def test_validate_lookahead_bias_no_matching_timestamps(self):
        """Test lookahead bias validation with no matching timestamps"""
        mismatched_signals = pl.DataFrame({
            'timestamp': ['2020-02-01'],  # Different date
            'signal': [1]
        })
        
        result = self.validator.validate_lookahead_bias(self.valid_data, mismatched_signals)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "No matching timestamps" in result.message
    
    def test_validate_lookahead_bias_insufficient_data(self):
        """Test lookahead bias validation with insufficient signal data"""
        few_signals = self.signals.slice(0, 1)  # Only one signal
        
        result = self.validator.validate_lookahead_bias(self.valid_data, few_signals)
        
        assert result.passed is True
        assert result.severity == "info"
        assert "Insufficient signal data" in result.message
    
    def test_validate_realistic_costs_disabled(self):
        """Test realistic costs validation when disabled"""
        validator = BiasValidator(warn_on_low_costs=False)
        result = validator.validate_realistic_costs(0.1, 0.0001)  # Very low costs
        
        assert result.name == "Realistic Costs"
        assert result.passed is True
        assert result.severity == "info"
        assert "disabled" in result.message.lower()
    
    def test_validate_realistic_costs_low_commission(self):
        """Test realistic costs validation with low commission"""
        result = self.validator.validate_realistic_costs(0.5, 0.001)  # Very low commission
        
        assert result.passed is False
        assert result.severity == "warning"
        assert "Very low commission" in result.message
    
    def test_validate_realistic_costs_high_commission(self):
        """Test realistic costs validation with high commission"""
        result = self.validator.validate_realistic_costs(15.0, 0.001)  # High commission
        
        assert result.passed is False
        assert result.severity == "warning"
        assert "High commission" in result.message
    
    def test_validate_realistic_costs_low_slippage(self):
        """Test realistic costs validation with low slippage"""
        result = self.validator.validate_realistic_costs(5.0, 0.0001)  # Very low slippage
        
        assert result.passed is False
        assert result.severity == "warning"
        assert "Very low slippage" in result.message
    
    def test_validate_realistic_costs_high_slippage(self):
        """Test realistic costs validation with high slippage"""
        result = self.validator.validate_realistic_costs(5.0, 0.01)  # High slippage
        
        assert result.passed is False
        assert result.severity == "warning"
        assert "High slippage" in result.message
    
    def test_validate_realistic_costs_good(self):
        """Test realistic costs validation with good values"""
        result = self.validator.validate_realistic_costs(5.0, 0.001)  # Realistic values
        
        assert result.passed is True
        assert result.severity == "info"
        assert "look realistic" in result.message.lower()
    
    def test_validate_vs_buy_and_hold_success(self):
        """Test buy-and-hold comparison validation success"""
        strategy_returns = [0.01, 0.02, -0.01, 0.03] * 10  # 40 days total
        buy_hold_returns = [0.005, 0.01, -0.005, 0.015] * 10  # 40 days total
        
        result = self.validator.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns)
        
        assert result.name == "Buy-and-Hold Comparison"
        assert result.passed is True
        assert result.severity == "info"
        assert "outperformed" in result.message.lower()
    
    def test_validate_vs_buy_and_hold_underperformance(self):
        """Test buy-and-hold comparison with underperformance"""
        strategy_returns = [0.001, 0.002, -0.001, 0.003] * 10  # Lower returns
        buy_hold_returns = [0.01, 0.02, -0.01, 0.03] * 10  # Higher returns
        
        result = self.validator.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns)
        
        assert result.name == "Buy-and-Hold Comparison"
        assert result.passed is True
        assert result.severity == "info"
        assert "underperformed" in result.message.lower()
    
    def test_validate_vs_buy_and_hold_extreme_underperformance(self):
        """Test buy-and-hold comparison with extreme underperformance"""
        strategy_returns = [-0.5] + [0.001] * 30  # -50% return
        buy_hold_returns = [0.01] * 31  # Small positive returns
        
        result = self.validator.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns)
        
        assert result.passed is False  # Should trigger warning
        assert result.severity == "warning"
        assert "underperformed" in result.message.lower()
    
    def test_validate_vs_buy_and_hold_different_lengths(self):
        """Test buy-and-hold comparison with different length arrays"""
        strategy_returns = [0.01, 0.02]
        buy_hold_returns = [0.01]  # Different length
        
        result = self.validator.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns)
        
        assert result.passed is False
        assert result.severity == "error"
        assert "different lengths" in result.message
    
    def test_validate_vs_buy_and_hold_insufficient_data(self):
        """Test buy-and-hold comparison with insufficient data"""
        strategy_returns = [0.01]  # Only one data point
        buy_hold_returns = [0.01]
        
        result = self.validator.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns)
        
        assert result.passed is True
        assert result.severity == "info"
        assert "Insufficient data" in result.message
    
    def test_validate_trade_frequency_success(self):
        """Test trade frequency validation with good frequency"""
        result = self.validator.validate_trade_frequency(self.trades, 252)  # 252 trading days
        
        assert result.name == "Trade Frequency"
        assert result.passed is True
        assert result.severity == "info"
        assert "2 trades" in result.message
        assert "Low" in result.message
    
    def test_validate_trade_frequency_no_trades(self):
        """Test trade frequency validation with no trades"""
        result = self.validator.validate_trade_frequency([], 252)
        
        assert result.name == "Trade Frequency"
        assert result.passed is False
        assert result.severity == "warning"
        assert "No trades executed" in result.message
    
    def test_validate_trade_frequency_very_high(self):
        """Test trade frequency validation with very high frequency"""
        many_trades = [{'timestamp': f'2020-01-{i:02d}', 'side': 'BUY', 'price': 100.0, 'quantity': 100} 
                       for i in range(1, 31)]  # 30 trades in 30 days
        
        result = self.validator.validate_trade_frequency(many_trades, 30)
        
        assert result.name == "Trade Frequency"
        assert result.passed is True
        assert result.severity == "warning"  # Should warn about high frequency
        assert "Very High" in result.message
    
    def test_validate_all_success(self):
        """Test running all validations successfully"""
        results = self.validator.validate_all(
            data=self.valid_data,
            signals=self.signals,
            trades=self.trades,
            commission_bps=5.0,
            slippage_pct=0.001,
            strategy_returns=[0.01, 0.02, -0.01],
            buy_hold_returns=[0.005, 0.01, -0.005]
        )
        
        assert len(results) == 5  # All 5 validations
        self.validator.results = results  # Set results for summary test
        
        # Most should pass with good test data
        passed_count = sum(1 for r in results if r.passed)
        assert passed_count >= 4  # At least 4 should pass
    
    def test_validate_all_with_errors(self):
        """Test running all validations with some errors"""
        bad_data = self.valid_data.with_columns([
            pl.lit(None).alias('close')
        ])
        
        results = self.validator.validate_all(
            data=bad_data,
            signals=self.signals,
            trades=[],
            commission_bps=0.5,  # Very low commission
            slippage_pct=0.0001  # Very low slippage
        )
        
        assert len(results) == 4  # Without strategy returns comparison
        self.validator.results = results
        
        # Should have some failures
        failed_count = sum(1 for r in results if not r.passed)
        assert failed_count >= 2  # Data quality and costs should fail
    
    def test_get_summary(self):
        """Test getting validation summary"""
        # Set up some test results
        self.validator.results = [
            ValidationResult("Test1", True, "OK", "info"),
            ValidationResult("Test2", False, "Warning", "warning"),
            ValidationResult("Test3", False, "Error", "error"),
            ValidationResult("Test4", True, "OK", "info"),
        ]
        
        summary = self.validator.get_summary()
        
        assert summary['total'] == 4
        assert summary['passed'] == 2
        assert summary['warnings'] == 1
        assert summary['errors'] == 1
        assert summary['infos'] == 2
        assert summary['has_critical_issues'] is True
    
    def test_get_summary_empty(self):
        """Test getting summary with no results"""
        summary = self.validator.get_summary()
        
        assert summary['total'] == 0
        assert summary['passed'] == 0
        assert summary['warnings'] == 0
        assert summary['errors'] == 0
        assert summary['infos'] == 0
        # has_critical_issues should be False when no errors
        assert summary.get('has_critical_issues', False) is False
    
    def test_print_results_no_results(self):
        """Test printing results with no validation results"""
        # This should not raise an exception
        self.validator.print_results()  # Should handle empty results gracefully


class TestValidationIntegration:
    """Integration tests for validation module"""
    
    def test_real_world_validation_scenario(self):
        """Test validation with realistic backtest data"""
        validator = BiasValidator()
        
        # Create realistic data
        dates = [f'2020-01-{i:02d}' for i in range(1, 32)]  # January 2020
        n_days = len(dates)
        
        # Simulate price movement
        import random
        random.seed(42)
        prices = [100.0]
        for i in range(1, n_days):
            daily_return = random.gauss(0.001, 0.02)  # 0.1% mean, 2% volatility
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Create realistic OHLCV data
        data = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(random.gauss(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(random.gauss(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [random.randint(500000, 2000000) for _ in range(n_days)]
        })
        
        # Create some trading signals
        signals = pl.DataFrame({
            'timestamp': dates,
            'signal': [1 if i % 10 == 0 else (-1 if i % 15 == 0 else 0) for i in range(n_days)]
        })
        
        # Create some trades
        trade_days = [0, 10, 15, 20, 25]
        trades = [
            {
                'timestamp': dates[i],
                'side': 'BUY' if i in [0, 15] else 'SELL',
                'price': prices[i],
                'quantity': 100,
                'commission': 5.0,
                'slippage': 0.1
            }
            for i in trade_days
        ]
        
        # Run all validations
        results = validator.validate_all(
            data=data,
            signals=signals,
            trades=trades,
            commission_bps=5.0,
            slippage_pct=0.001,
            strategy_returns=[random.gauss(0.001, 0.02) for _ in range(n_days)],
            buy_hold_returns=[random.gauss(0.0008, 0.018) for _ in range(n_days)]
        )
        
        # Should complete without errors
        assert len(results) == 5
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Get summary
        validator.results = results
        summary = validator.get_summary()
        
        # Most validations should pass with realistic data
        assert summary['passed'] >= 3
        assert summary['errors'] == 0  # Should not have critical errors with good data