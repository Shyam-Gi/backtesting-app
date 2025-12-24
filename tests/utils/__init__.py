"""Test utilities for backtesting system."""

import numpy as np
import polars as pl
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


def assert_dataframe_equal(df1: pl.DataFrame, df2: pl.DataFrame, 
                           check_dtype: bool = True, 
                           tolerance: float = 1e-10) -> None:
    """Assert two DataFrames are equal with proper tolerance for floats.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame  
        check_dtype: Whether to check data types
        tolerance: Tolerance for float comparisons
        
    Raises:
        AssertionError: If DataFrames are not equal
    """
    assert df1.shape == df2.shape, f"Shapes differ: {df1.shape} vs {df2.shape}"
    assert list(df1.columns) == list(df2.columns), f"Columns differ: {df1.columns} vs {df2.columns}"
    
    for col in df1.columns:
        if df1[col].dtype in [pl.Float32, pl.Float64]:
            # Use numpy's allclose for float comparisons
            diff = np.abs(df1[col].to_numpy() - df2[col].to_numpy())
            assert np.all(diff <= tolerance), f"Column '{col}' differs beyond tolerance {tolerance}"
        else:
            # Exact comparison for non-float columns
            assert df1[col].equals(df2[col]), f"Column '{col}' differs"


def assert_trades_equal(trades1: List[Dict], trades2: List[Dict], 
                       tolerance: float = 1e-10) -> None:
    """Assert two lists of trades are equal.
    
    Args:
        trades1: First list of trades
        trades2: Second list of trades
        tolerance: Tolerance for float comparisons
    """
    assert len(trades1) == len(trades2), f"Trade counts differ: {len(trades1)} vs {len(trades2)}"
    
    for i, (trade1, trade2) in enumerate(zip(trades1, trades2)):
        assert set(trade1.keys()) == set(trade2.keys()), f"Trade keys differ at index {i}"
        
        for key in trade1.keys():
            if isinstance(trade1[key], float):
                assert abs(trade1[key] - trade2[key]) <= tolerance, \
                    f"Trade {key} differs at index {i}: {trade1[key]} vs {trade2[key]}"
            else:
                assert trade1[key] == trade2[key], \
                    f"Trade {key} differs at index {i}: {trade1[key]} vs {trade2[key]}"


def generate_ohlcv_with_trend(start_date: str, end_date: str, 
                             trend: float = 0.001, volatility: float = 0.02,
                             initial_price: float = 100.0,
                             seed: Optional[int] = None) -> pl.DataFrame:
    """Generate OHLCV data with specified trend and volatility.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        trend: Daily drift (positive for uptrend, negative for downtrend)
        volatility: Daily volatility (standard deviation)
        initial_price: Starting price
        seed: Random seed for reproducibility
        
    Returns:
        Polars DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    # Generate price series with trend
    returns = np.random.normal(trend, volatility, n_days)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Generate OHLC from close prices
    intraday_vol = volatility * 0.5
    
    data = {
        "timestamp": dates,
        "open": prices * np.exp(np.random.normal(0, intraday_vol, n_days)),
        "close": prices,
        "high": prices * np.exp(np.abs(np.random.normal(0, intraday_vol, n_days))),
        "low": prices * np.exp(-np.abs(np.random.normal(0, intraday_vol, n_days))),
        "volume": np.random.lognormal(14.0, 0.5, n_days).astype(int)
    }
    
    return pl.from_pandas(pd.DataFrame(data))


def calculate_manual_returns(prices: List[float]) -> List[float]:
    """Calculate manual returns from price series.
    
    Args:
        prices: List of prices
        
    Returns:
        List of returns (excluding first return which is NaN)
    """
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    return returns


def calculate_manual_sharpe(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate manual Sharpe ratio.
    
    Args:
        returns: List of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_manual_max_drawdown(nav_history: List[float]) -> float:
    """Calculate manual maximum drawdown.
    
    Args:
        nav_history: List of NAV values
        
    Returns:
        Maximum drawdown as a percentage (negative value)
    """
    if not nav_history:
        return 0.0
    
    peak = nav_history[0]
    max_dd = 0.0
    
    for nav in nav_history:
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    return max_dd


def validate_ohlcv_data(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate OHLCV data and return validation results.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check required columns
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    
    # Check for null values
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null values")
    
    # Check OHLC relationships
    if "high" in df.columns and "low" in df.columns and "open" in df.columns and "close" in df.columns:
        # High should be >= Open and Close
        high_violations = df.filter(
            (pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))
        ).height
        if high_violations > 0:
            issues.append(f"High < Open/Close in {high_violations} rows")
        
        # Low should be <= Open and Close
        low_violations = df.filter(
            (pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))
        ).height
        if low_violations > 0:
            issues.append(f"Low > Open/Close in {low_violations} rows")
    
    # Check for negative values
    if "volume" in df.columns:
        negative_volume = df.filter(pl.col("volume") < 0).height
        if negative_volume > 0:
            issues.append(f"Negative volume in {negative_volume} rows")
    
    # Check for duplicate timestamps
    if "timestamp" in df.columns:
        duplicates = df.height - df.select("timestamp").n_unique()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
    
    # Check data frequency (warning if not daily)
    if "timestamp" in df.columns and df.height >= 2:
        time_diffs = df.sort("timestamp").select(
            pl.col("timestamp").diff().alias("diff")
        ).filter(pl.col("diff").is_not_null())
        
        if time_diffs.height > 0:
            median_diff = time_diffs.select(pl.col("diff").median()).item()
            if median_diff.days != 1:
                warnings.append(f"Data frequency appears to be {median_diff.days}, not daily")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "rows": df.height,
        "columns": df.columns
    }


def create_test_strategy_signals(df: pl.DataFrame, signal_pattern: List[int]) -> pl.DataFrame:
    """Create test signals based on a pattern.
    
    Args:
        df: OHLCV DataFrame
        signal_pattern: List of signals to repeat (-1, 0, 1)
        
    Returns:
        DataFrame with signals column added
    """
    n_rows = df.height
    signals = []
    
    for i in range(n_rows):
        signal = signal_pattern[i % len(signal_pattern)]
        signals.append(signal)
    
    return df.with_columns(
        pl.Series("signal", signals)
    )