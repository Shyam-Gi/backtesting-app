"""Tests for Streamlit app integration.

Tests for DataFrame handling, plotting functions, and UI components.
"""

import pytest
import polars as pl
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# Import app functions for testing
import sys
sys.path.append('.')
from app import plot_equity_curve, plot_drawdown


class TestAppIntegration:
    """Test suite for Streamlit app integration."""
    
    @pytest.fixture
    def sample_nav_history(self):
        """Create sample NAV history as Polars DataFrame."""
        return pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
            'cash': [50000.0, 49500.0, 49000.0, 50750.0, 51500.0],
            'position_value': [50000.0, 51500.0, 53000.0, 50750.0, 51500.0],
            'total_value': [100000.0, 101000.0, 102000.0, 101500.0, 103000.0],
            'position_quantity': [500, 515, 530, 507, 515],
            'unrealized_pnl': [0.0, 1500.0, 3000.0, 750.0, 1500.0],
            'realized_pnl': [0.0, 0.0, 0.0, 0.0, 0.0],
            'total_pnl': [0.0, 1500.0, 3000.0, 750.0, 1500.0]
        })
    
    @pytest.fixture
    def empty_nav_history(self):
        """Create empty NAV history DataFrame."""
        return pl.DataFrame(schema={
            'timestamp': pl.Utf8,
            'cash': pl.Float64,
            'position_value': pl.Float64,
            'total_value': pl.Float64,
            'position_quantity': pl.Int64,
            'unrealized_pnl': pl.Float64,
            'realized_pnl': pl.Float64,
            'total_pnl': pl.Float64
        })
    
    def test_plot_equity_curve_with_data(self, sample_nav_history):
        """Test equity curve plotting with valid data."""
        # Should not raise exception
        fig = plot_equity_curve(sample_nav_history)
        
        # Check that figure is created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Check that the trace uses correct column
        trace = fig.data[0]
        assert len(trace.y) == len(sample_nav_history)
        assert all(y > 0 for y in trace.y)  # Portfolio values should be positive
    
    def test_plot_equity_curve_with_empty_data(self, empty_nav_history):
        """Test equity curve plotting with empty data."""
        # Should return empty figure without error
        fig = plot_equity_curve(empty_nav_history)
        
        # Check that empty figure is returned
        assert fig is not None
        assert len(fig.data) == 0
    
    def test_plot_drawdown_with_data(self, sample_nav_history):
        """Test drawdown plotting with valid data."""
        # Should not raise exception
        fig = plot_drawdown(sample_nav_history)
        
        # Check that figure is created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        
        # Check that drawdown values are calculated
        trace = fig.data[0]
        assert len(trace.y) == len(sample_nav_history)
        # Drawdown should be percentage values (can be negative or zero)
        assert all(y <= 100 for y in trace.y)  # Drawdown shouldn't exceed 100%
    
    def test_plot_drawdown_with_empty_data(self, empty_nav_history):
        """Test drawdown plotting with empty data."""
        # Should return empty figure without error
        fig = plot_drawdown(empty_nav_history)
        
        # Check that empty figure is returned
        assert fig is not None
        assert len(fig.data) == 0
    
    def test_dataframe_boolean_evaluation(self, sample_nav_history, empty_nav_history):
        """Test proper DataFrame boolean evaluation."""
        # Test with data
        assert not sample_nav_history.is_empty()
        
        # Test with empty data
        assert empty_nav_history.is_empty()
        
        # These should not raise TypeError
        if not sample_nav_history.is_empty():
            assert True
        
        if empty_nav_history.is_empty():
            assert True
    
    def test_polars_to_pandas_conversion(self, sample_nav_history):
        """Test Polars to pandas conversion in plotting functions."""
        # Test that conversion works without error
        pandas_df = sample_nav_history.to_pandas()
        
        assert isinstance(pandas_df, pd.DataFrame)
        assert len(pandas_df) == len(sample_nav_history)
        assert 'total_value' in pandas_df.columns
        assert 'timestamp' in pandas_df.columns
    
    def test_column_name_mapping(self, sample_nav_history):
        """Test that plotting functions use correct column names."""
        # Test equity curve uses total_value
        pandas_df = sample_nav_history.to_pandas()
        assert 'total_value' in pandas_df.columns
        
        # Test drawdown uses total_value for nav calculation
        fig = plot_drawdown(sample_nav_history)
        trace = fig.data[0]
        assert len(trace.y) > 0  # Drawdown values should be calculated


class TestAppEdgeCases:
    """Test edge cases for app integration."""
    
    def test_single_data_point(self):
        """Test plotting with single data point."""
        single_point_df = pl.DataFrame({
            'timestamp': ['2020-01-01'],
            'cash': [50000.0],
            'position_value': [50000.0],
            'total_value': [100000.0],
            'position_quantity': [500],
            'unrealized_pnl': [0.0],
            'realized_pnl': [0.0],
            'total_pnl': [0.0]
        })
        
        # Should handle single point without error
        fig_equity = plot_equity_curve(single_point_df)
        fig_drawdown = plot_drawdown(single_point_df)
        
        assert fig_equity is not None
        assert fig_drawdown is not None
    
    def test_negative_portfolio_values(self):
        """Test handling of negative portfolio values."""
        negative_df = pl.DataFrame({
            'timestamp': ['2020-01-01', '2020-01-02'],
            'cash': [50000.0, 40000.0],
            'position_value': [50000.0, 45000.0],
            'total_value': [100000.0, 85000.0],  # Decreasing value
            'position_quantity': [500, 450],
            'unrealized_pnl': [0.0, -5000.0],
            'realized_pnl': [0.0, 0.0],
            'total_pnl': [0.0, -5000.0]
        })
        
        # Should handle negative P&L without error
        fig_equity = plot_equity_curve(negative_df)
        fig_drawdown = plot_drawdown(negative_df)
        
        assert fig_equity is not None
        assert fig_drawdown is not None
    
    def test_zero_portfolio_values(self):
        """Test handling of zero portfolio values."""
        zero_df = pl.DataFrame({
            'timestamp': ['2020-01-01'],
            'cash': [0.0],
            'position_value': [0.0],
            'total_value': [0.0],
            'position_quantity': [0],
            'unrealized_pnl': [0.0],
            'realized_pnl': [0.0],
            'total_pnl': [0.0]
        })
        
        # Should handle zero values without error
        fig_equity = plot_equity_curve(zero_df)
        fig_drawdown = plot_drawdown(zero_df)
        
        assert fig_equity is not None
        assert fig_drawdown is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])