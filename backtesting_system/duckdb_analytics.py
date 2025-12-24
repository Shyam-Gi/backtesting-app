"""DuckDB analytics integration for fast metrics calculation.

Provides high-performance analytical queries using DuckDB.
Optimizes calculations for large datasets with millions of rows.
"""

import polars as pl
import duckdb
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

from .metrics import PerformanceMetrics
from .simulator import Trade


class DuckDBAnalytics:
    """Fast analytics engine using DuckDB for performance metrics.
    
    Provides optimized calculations for large datasets.
    Handles complex analytical queries with sub-second response times.
    
    Parameters:
        memory_limit: Memory limit for DuckDB (default: '1GB')
        
    Example:
        analytics = DuckDBAnalytics(memory_limit='2GB')
        metrics = analytics.calculate_metrics(nav_history, trades)
    """
    
    def __init__(self, memory_limit: str = '1GB'):
        self.memory_limit = memory_limit
        self._conn = None
        
    def __enter__(self):
        """Context manager entry."""
        self._conn = duckdb.connect(':memory:')
        self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
        self._conn.execute("SET enable_progress_bar=false")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    @property
    def conn(self):
        """Get DuckDB connection, creating if needed."""
        if self._conn is None:
            self._conn = duckdb.connect(':memory:')
            self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
            self._conn.execute("SET enable_progress_bar=false")
        return self._conn
    
    def calculate_metrics(
        self,
        nav_history: pl.DataFrame,
        trades: List[Trade],
        benchmark_data: Optional[pl.DataFrame] = None,
        use_analytics_db: bool = True
    ) -> PerformanceMetrics:
        """Calculate performance metrics using DuckDB for speed.
        
        Args:
            nav_history: Daily portfolio values
            trades: List of executed trades
            benchmark_data: Optional benchmark data
            use_analytics_db: Whether to use DuckDB (fallback to Python if False)
            
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        if not use_analytics_db or nav_history.height == 0:
            # Fallback to regular calculation
            from .metrics import MetricsCalculator
            calc = MetricsCalculator()
            return calc.calculate(nav_history, trades, benchmark_data)
        
        # Register DataFrames with DuckDB
        self.conn.register('nav', nav_history)
        
        if trades:
            trades_df = pl.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'side': t.side,
                    'price': t.price,
                    'quantity': t.quantity,
                    'commission': t.commission,
                    'slippage': t.slippage,
                    'total_cost': t.total_cost
                }
                for t in trades
            ])
            self.conn.register('trades', trades_df)
        
        if benchmark_data is not None:
            self.conn.register('benchmark', benchmark_data)
        
        # Calculate metrics using optimized SQL queries
        returns_metrics = self._calculate_returns_duckdb()
        risk_metrics = self._calculate_risk_metrics_duckdb(returns_metrics)
        trade_metrics = self._calculate_trade_statistics_duckdb(trades)
        
        # Calculate risk-adjusted returns
        risk_adjusted = self._calculate_risk_adjusted_returns_duckdb(
            returns_metrics, risk_metrics, benchmark_data
        )
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics_duckdb(returns_metrics, trades)
        
        return PerformanceMetrics(
            # Returns
            total_return=returns_metrics['total_return'],
            cagr=returns_metrics['cagr'],
            annualized_return=returns_metrics['annualized_return'],
            daily_return_mean=returns_metrics['daily_return_mean'],
            daily_return_std=returns_metrics['daily_return_std'],
            
            # Risk Metrics
            volatility=risk_metrics['volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            max_drawdown_duration=int(risk_metrics['max_drawdown_duration']),
            calmar_ratio=risk_metrics['calmar_ratio'],
            
            # Risk-Adjusted Returns
            sharpe_ratio=risk_adjusted['sharpe_ratio'],
            sortino_ratio=risk_adjusted['sortino_ratio'],
            information_ratio=risk_adjusted['information_ratio'],
            beta=risk_adjusted['beta'],
            alpha=risk_adjusted['alpha'],
            
            # Trade Statistics
            win_rate=trade_metrics['win_rate'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            profit_factor=trade_metrics['profit_factor'],
            expectancy=trade_metrics['expectancy'],
            trade_count=trade_metrics['trade_count'],
            avg_holding_period=trade_metrics['avg_holding_period'],
            
            # Efficiency
            turnover=efficiency_metrics['turnover'],
            var_95=efficiency_metrics['var_95'],
            skewness=efficiency_metrics['skewness'],
            kurtosis=efficiency_metrics['kurtosis']
        )
    
    def _calculate_returns_duckdb(self) -> Dict[str, float]:
        """Calculate return metrics using DuckDB."""
        query = """
        WITH returns AS (
            SELECT 
                timestamp,
                total_value,
                LAG(total_value) OVER (ORDER BY timestamp) as prev_value,
                (total_value / LAG(total_value) OVER (ORDER BY timestamp) - 1) as daily_return
            FROM nav
            WHERE prev_value IS NOT NULL
        ),
        stats AS (
            SELECT 
                AVG(daily_return) as daily_return_mean,
                STDDEV(daily_return) as daily_return_std,
                COUNT(*) as trading_days,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM returns
        ),
        bookends AS (
            SELECT 
                (SELECT total_value FROM nav ORDER BY timestamp LIMIT 1) as initial_value,
                (SELECT total_value FROM nav ORDER BY timestamp DESC LIMIT 1) as final_value
        )
        SELECT 
            s.daily_return_mean,
            s.daily_return_std,
            (b.final_value - b.initial_value) / b.initial_value as total_return,
            b.final_value / b.initial_value as final_to_initial_ratio,
            s.trading_days
        FROM stats s, bookends b
        """
        
        result = self.conn.execute(query).fetchdf()
        row = result.iloc[0]
        
        # Calculate CAGR and annualized return
        # Parse dates to calculate years
        start_ts = self.conn.execute("SELECT MIN(timestamp) FROM nav").fetchone()[0]
        end_ts = self.conn.execute("SELECT MAX(timestamp) FROM nav").fetchone()[0]
        
        try:
            start_date = datetime.strptime(str(start_ts).split()[0], '%Y-%m-%d')
            end_date = datetime.strptime(str(end_ts).split()[0], '%Y-%m-%d')
            years = (end_date - start_date).days / 365.25
            cagr = (row['final_to_initial_ratio'] ** (1 / years) - 1) if years > 0 else 0
        except:
            cagr = 0
        
        annualized_return = row['daily_return_mean'] * 252  # Trading days per year
        
        return {
            'total_return': float(row['total_return']),
            'cagr': float(cagr),
            'annualized_return': float(annualized_return),
            'daily_return_mean': float(row['daily_return_mean']),
            'daily_return_std': float(row['daily_return_std'])
        }
    
    def _calculate_risk_metrics_duckdb(self, returns_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk metrics using DuckDB."""
        query = """
        WITH peaks AS (
            SELECT 
                timestamp,
                total_value,
                MAX(total_value) OVER (ORDER BY timestamp) as peak_value
            FROM nav
        ),
        drawdowns AS (
            SELECT 
                timestamp,
                (peak_value - total_value) / peak_value as drawdown,
                CASE 
                    WHEN LAG((peak_value - total_value) / peak_value) OVER (ORDER BY timestamp) > 0 
                    THEN LAG((peak_value - total_value) / peak_value) OVER (ORDER BY timestamp) + 1
                    ELSE 1
                END as drawdown_streak
            FROM peaks
        ),
        drawdown_groups AS (
            SELECT 
                timestamp,
                drawdown,
                SUM(CASE WHEN drawdown > 0 THEN 1 ELSE 0 END) 
                    OVER (ORDER BY timestamp ORDER BY drawdown_streak DESC) as dd_group
            FROM drawdowns
        )
        SELECT 
            MAX(drawdown) as max_drawdown,
            MAX(COUNT(*)) OVER (PARTITION BY dd_group) as max_drawdown_duration
        FROM drawdown_groups
        """
        
        result = self.conn.execute(query).fetchdf()
        row = result.iloc[0]
        
        volatility = returns_metrics['daily_return_std'] * np.sqrt(252)
        calmar_ratio = returns_metrics['cagr'] / row['max_drawdown'] if row['max_drawdown'] > 0 else 0
        
        return {
            'volatility': float(volatility),
            'max_drawdown': float(row['max_drawdown']),
            'max_drawdown_duration': int(row['max_drawdown_duration']),
            'calmar_ratio': float(calmar_ratio)
        }
    
    def _calculate_trade_statistics_duckdb(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate trade statistics using DuckDB."""
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'trade_count': 0,
                'avg_holding_period': 0.0
            }
        
        # Complex trade P&L calculation using window functions
        query = """
        WITH trade_pairs AS (
            SELECT 
                t1.timestamp as buy_timestamp,
                t1.price as buy_price,
                t1.quantity as buy_quantity,
                t2.timestamp as sell_timestamp,
                t2.price as sell_price,
                t1.quantity as sell_quantity,
                t2.commission as sell_commission,
                (sell_price - buy_price) * t1.quantity - sell_commission as pnl,
                DATEDIFF('day', t1.timestamp, t2.timestamp) as holding_days
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp, side) as rn 
                FROM trades WHERE side = 'BUY'
            ) t1
            JOIN (
                SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp, side) as rn 
                FROM trades WHERE side = 'SELL'
            ) t2 ON t1.rn = t2.rn
        ),
        trade_stats AS (
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss,
                SUM(CASE WHEN pnl > 0 THEN pnl END) as total_wins,
                SUM(CASE WHEN pnl <= 0 THEN ABS(pnl) END) as total_losses,
                AVG(pnl) as expectancy,
                AVG(holding_days) as avg_holding_period
            FROM trade_pairs
        )
        SELECT 
            total_trades,
            CAST(winning_trades AS FLOAT) / total_trades as win_rate,
            COALESCE(avg_win, 0) as avg_win,
            COALESCE(avg_loss, 0) as avg_loss,
            CASE WHEN total_losses > 0 THEN total_wins / total_losses ELSE 0 END as profit_factor,
            COALESCE(expectancy, 0) as expectancy,
            COALESCE(avg_holding_period, 0) as avg_holding_period
        FROM trade_stats
        """
        
        try:
            result = self.conn.execute(query).fetchdf()
            row = result.iloc[0]
            
            return {
                'win_rate': float(row['win_rate']),
                'avg_win': float(row['avg_win']),
                'avg_loss': float(row['avg_loss']),
                'profit_factor': float(row['profit_factor']),
                'expectancy': float(row['expectancy']),
                'trade_count': int(row['total_trades']),
                'avg_holding_period': float(row['avg_holding_period'])
            }
        except:
            # Fallback for complex queries
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'trade_count': len(trades),
                'avg_holding_period': 0.0
            }
    
    def _calculate_risk_adjusted_returns_duckdb(
        self, 
        returns_metrics: Dict[str, float], 
        risk_metrics: Dict[str, float],
        benchmark_data: Optional[pl.DataFrame]
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if benchmark_data is None:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'information_ratio': 0.0,
                'beta': 0.0,
                'alpha': 0.0
            }
        
        # Calculate benchmark returns and correlations
        query = """
        WITH portfolio_returns AS (
            SELECT 
                timestamp,
                (total_value / LAG(total_value) OVER (ORDER BY timestamp) - 1) as portfolio_return
            FROM nav
            WHERE LAG(total_value) OVER (ORDER BY timestamp) IS NOT NULL
        ),
        benchmark_returns AS (
            SELECT 
                timestamp,
                (close / LAG(close) OVER (ORDER BY timestamp) - 1) as benchmark_return
            FROM benchmark
            WHERE LAG(close) OVER (ORDER BY timestamp) IS NOT NULL
        ),
        joined_returns AS (
            SELECT 
                p.portfolio_return,
                b.benchmark_return
            FROM portfolio_returns p
            JOIN benchmark_returns b ON p.timestamp = b.timestamp
        ),
        return_stats AS (
            SELECT 
                AVG(portfolio_return) as avg_portfolio_return,
                STDDEV(portfolio_return) as portfolio_std,
                AVG(benchmark_return) as avg_benchmark_return,
                STDDEV(benchmark_return) as benchmark_std,
                STDDEV(portfolio_return - benchmark_return) as tracking_error,
                COVAR(portfolio_return, benchmark_return) as covariance,
                VAR(benchmark_return) as benchmark_variance
            FROM joined_returns
        )
        SELECT 
            r.avg_portfolio_return * 252 as portfolio_return_annual,
            r.portfolio_std * SQRT(252) as portfolio_volatility,
            r.avg_benchmark_return * 252 as benchmark_return_annual,
            r.tracking_error * SQRT(252) as tracking_error_annual,
            CASE WHEN r.benchmark_variance > 0 THEN r.covariance / r.benchmark_variance ELSE 0 END as beta,
            r.covariance as covariance
        FROM return_stats r
        """
        
        try:
            result = self.conn.execute(query).fetchdf()
            row = result.iloc[0]
            
            risk_free_rate = 0.02  # 2% risk-free rate
            portfolio_excess = row['portfolio_return_annual'] - risk_free_rate
            
            sharpe_ratio = portfolio_excess / row['portfolio_volatility'] if row['portfolio_volatility'] > 0 else 0
            
            # Sortino ratio (simplified)
            negative_returns_query = """
            SELECT STDDEV(portfolio_return) as downside_std
            FROM (
                SELECT portfolio_return
                FROM portfolio_returns
                WHERE portfolio_return < 0
            )
            """
            downside_result = self.conn.execute(negative_returns_query).fetchdf()
            downside_std = downside_result.iloc[0]['downside_std'] if not downside_result.empty else 0
            downside_volatility = downside_std * np.sqrt(252)
            sortino_ratio = portfolio_excess / downside_volatility if downside_volatility > 0 else 0
            
            # Information ratio
            portfolio_excess_return = row['portfolio_return_annual'] - row['benchmark_return_annual']
            information_ratio = portfolio_excess_return / row['tracking_error_annual'] if row['tracking_error_annual'] > 0 else 0
            
            # Alpha (CAPM)
            alpha = row['portfolio_return_annual'] - (risk_free_rate + row['beta'] * (row['benchmark_return_annual'] - risk_free_rate))
            
            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'information_ratio': float(information_ratio),
                'beta': float(row['beta']),
                'alpha': float(alpha)
            }
        except:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'information_ratio': 0.0,
                'beta': 0.0,
                'alpha': 0.0
            }
    
    def _calculate_efficiency_metrics_duckdb(self, returns_metrics: Dict[str, float], trades: List[Trade]) -> Dict[str, float]:
        """Calculate efficiency and distribution metrics."""
        # VaR calculation using quantile
        var_query = """
        WITH returns AS (
            SELECT (total_value / LAG(total_value) OVER (ORDER BY timestamp) - 1) as daily_return
            FROM nav
            WHERE LAG(total_value) OVER (ORDER BY timestamp) IS NOT NULL
        )
        SELECT 
            QUANTILE(daily_return, 0.05) as var_95,
            SKEWNESS(daily_return) as skewness,
            KURTOSIS(daily_return) as kurtosis
        FROM returns
        """
        
        try:
            var_result = self.conn.execute(var_query).fetchdf()
            var_row = var_result.iloc[0]
            
            # Turnover calculation
            if trades:
                total_volume = sum(trade.quantity * trade.price for trade in trades)
                avg_portfolio_value = sum(trade.price * trade.quantity for trade in trades) / len(trades) if trades else 0
                turnover = total_volume / avg_portfolio_value if avg_portfolio_value > 0 else 0
            else:
                turnover = 0.0
            
            return {
                'turnover': float(turnover),
                'var_95': float(var_row['var_95']),
                'skewness': float(var_row['skewness'] if var_row['skewness'] is not None else 0),
                'kurtosis': float(var_row['kurtosis'] if var_row['kurtosis'] is not None else 0)
            }
        except:
            return {
                'turnover': 0.0,
                'var_95': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
    
    def run_performance_benchmark(
        self,
        nav_history: pl.DataFrame,
        trades: List[Trade],
        iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark DuckDB vs Python performance.
        
        Args:
            nav_history: Portfolio history data
            trades: Trade list
            iterations: Number of iterations for timing
            
        Returns:
            Performance timing results
        """
        import time
        
        # Benchmark Python calculation
        python_times = []
        for _ in range(iterations):
            start_time = time.time()
            from .metrics import MetricsCalculator
            calc = MetricsCalculator()
            calc.calculate(nav_history, trades)
            python_times.append(time.time() - start_time)
        
        # Benchmark DuckDB calculation
        duckdb_times = []
        with self as analytics:
            for _ in range(iterations):
                start_time = time.time()
                analytics.calculate_metrics(nav_history, trades)
                duckdb_times.append(time.time() - start_time)
        
        return {
            'python_avg': np.mean(python_times),
            'python_std': np.std(python_times),
            'duckdb_avg': np.mean(duckdb_times),
            'duckdb_std': np.std(duckdb_times),
            'speedup': np.mean(python_times) / np.mean(duckdb_times),
            'python_total': np.sum(python_times),
            'duckdb_total': np.sum(duckdb_times)
        }