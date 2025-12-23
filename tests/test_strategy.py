"""Unit tests for trading strategies."""

import pytest
import polars as pl
import pandas as pd
from backtesting_system.strategy import BaseStrategy
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""

    def test_base_strategy_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_strategy_methods_exist(self):
        """Test that strategy methods exist and work."""
        strategy = SMAStrategy()
        
        # Test required methods exist
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'validate_signals')
        assert hasattr(strategy, 'get_params')
        assert hasattr(strategy, 'get_name')
        assert hasattr(strategy, '__repr__')

    def test_strategy_params(self):
        """Test strategy parameter handling."""
        strategy = SMAStrategy(fast_period=20, slow_period=100)
        
        params = strategy.get_params()
        assert params['fast_period'] == 20
        assert params['slow_period'] == 100
        assert strategy.fast_period == 20
        assert strategy.slow_period == 100

    def test_strategy_name(self):
        """Test strategy name method."""
        strategy = SMAStrategy()
        assert strategy.get_name() == "SMAStrategy"

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = SMAStrategy(fast_period=20, slow_period=100)
        repr_str = repr(strategy)
        assert "SMAStrategy" in repr_str
        assert "fast_period=20" in repr_str
        assert "slow_period=100" in repr_str


class TestSMAStrategy:
    """Test SMA Crossover strategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        # Create trending data: up first half, down second half
        close_prices = list(range(100, 115)) + list(range(115, 99, -1))
        
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [p - 0.5 for p in close_prices],
            'high': [p + 1.0 for p in close_prices],
            'low': [p - 1.0 for p in close_prices],
            'close': close_prices,
            'volume': [1000000] * len(dates)
        })
        return df

    def test_sma_strategy_signals(self, sample_data):
        """Test SMA strategy generates valid signals."""
        strategy = SMAStrategy(fast_period=5, slow_period=10)
        result = strategy.generate_signals(sample_data)
        
        # Check required columns exist
        assert 'fast_ma' in result.columns
        assert 'slow_ma' in result.columns
        assert 'signal' in result.columns
        
        # Check signal values are valid
        unique_signals = set(result.select('signal').to_series().to_list())
        assert unique_signals.issubset({1, -1, 0, None})

    def test_sma_strategy_crossover_logic(self, sample_data):
        """Test SMA crossover logic works correctly."""
        strategy = SMAStrategy(fast_period=5, slow_period=10)
        result = strategy.generate_signals(sample_data)
        
        # Validate signals
        strategy.validate_signals(result)
        
        # Check that fast MA is more responsive than slow MA
        fast_ma = result.select('fast_ma').to_series().to_list()
        slow_ma = result.select('slow_ma').to_series().to_list()
        
        # Fast MA should have more variance (more responsive)
        fast_variance = pl.Series(fast_ma).var()
        slow_variance = pl.Series(slow_ma).var()
        assert fast_variance > slow_variance

    def test_sma_strategy_periods(self, sample_data):
        """Test SMA strategy with different periods."""
        strategy = SMAStrategy(fast_period=3, slow_period=15)
        result = strategy.generate_signals(sample_data)
        
        # Should work with any valid periods
        assert len(result) == len(sample_data)
        assert 'signal' in result.columns


class TestMomentumStrategy:
    """Test Momentum strategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with clear momentum."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        # Create momentum data: gradual uptrend
        close_prices = [100 + i * 0.5 + (i % 5) for i in range(31)]
        
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [p - 0.2 for p in close_prices],
            'high': [p + 0.5 for p in close_prices],
            'low': [p - 0.5 for p in close_prices],
            'close': close_prices,
            'volume': [1000000] * len(close_prices)
        })
        return df

    def test_momentum_strategy_signals(self, sample_data):
        """Test momentum strategy generates valid signals."""
        strategy = MomentumStrategy(lookback_period=10, threshold=0.01)
        result = strategy.generate_signals(sample_data)
        
        # Check required columns exist
        assert 'momentum_pct' in result.columns
        assert 'signal' in result.columns
        
        # Check signal values are valid
        strategy.validate_signals(result)

    def test_momentum_calculation(self, sample_data):
        """Test momentum percentage calculation."""
        strategy = MomentumStrategy(lookback_period=10, threshold=0.01)
        result = strategy.generate_signals(sample_data)
        
        # Momentum should be NaN for first N-1 days
        momentum_vals = result.select('momentum_pct').to_series().to_list()
        assert momentum_vals[:9] == [None] * 9
        assert momentum_vals[10] is not None

    def test_momentum_thresholds(self, sample_data):
        """Test momentum threshold logic."""
        # Use very low threshold to ensure some signals
        strategy = MomentumStrategy(lookback_period=10, threshold=0.001)
        result = strategy.generate_signals(sample_data)
        
        signals = result.select('signal').to_series().to_list()
        # Should have some non-zero signals due to uptrend
        non_zero_signals = [s for s in signals if s not in [0, None]]
        assert len(non_zero_signals) > 0


class TestMeanReversionStrategy:
    """Test Mean Reversion strategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with mean reversion patterns."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        # Create oscillating data around mean
        base_price = 100
        oscillation = [5 * (i % 7 - 3) for i in range(31)]  # -15 to +15
        close_prices = [base_price + osc for osc in oscillation]
        
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [p - 0.2 for p in close_prices],
            'high': [p + 0.3 for p in close_prices],
            'low': [p - 0.3 for p in close_prices],
            'close': close_prices,
            'volume': [1000000] * len(close_prices)
        })
        return df

    def test_mean_reversion_signals(self, sample_data):
        """Test mean reversion strategy generates valid signals."""
        strategy = MeanReversionStrategy(ma_period=10, threshold=0.05)
        result = strategy.generate_signals(sample_data)
        
        # Check required columns exist
        assert 'ma' in result.columns
        assert 'distance' in result.columns
        assert 'signal' in result.columns
        
        # Check signal values are valid
        strategy.validate_signals(result)

    def test_distance_calculation(self, sample_data):
        """Test distance from moving average calculation."""
        strategy = MeanReversionStrategy(ma_period=10, threshold=0.05)
        result = strategy.generate_signals(sample_data)
        
        # Distance should be NaN for first N-1 days
        distance_vals = result.select('distance').to_series().to_list()
        assert distance_vals[:9] == [None] * 9
        assert distance_vals[10] is not None

    def test_mean_reversion_logic(self, sample_data):
        """Test mean reversion signal logic."""
        strategy = MeanReversionStrategy(ma_period=10, threshold=0.5)  # Large threshold
        result = strategy.generate_signals(sample_data)
        
        # With oscillating data and large threshold, should get various signals
        signals = result.select('signal').to_series().to_list()
        non_zero_signals = [s for s in signals if s not in [0, None]]
        # Should have some trading signals due to oscillation
        assert len(non_zero_signals) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])