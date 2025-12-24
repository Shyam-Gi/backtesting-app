"""Performance metrics calculation for backtesting.

Calculates 12+ standard trading metrics using vectorized operations.
Supports both Polars and DuckDB for fast analytics.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import duckdb
from datetime import datetime, timedelta

from .simulator import Trade


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Returns
    total_return: float
    cagr: float
    annualized_return: float
    daily_return_mean: float
    daily_return_std: float
    
    # Risk Metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    beta: float
    alpha: float
    
    # Trade Statistics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    trade_count: int
    avg_holding_period: float
    
    # Efficiency
    turnover: float
    var_95: float  # Value at Risk 95%
    skewness: float
    kurtosis: float


class MetricsCalculator:
    """Fast performance metrics calculator.
    
    Uses vectorized Polars operations and DuckDB for analytics.
    Calculates 12+ standard trading metrics.
    
    Parameters:
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 2%)
        
    Example:
        calc = MetricsCalculator(risk_free_rate=0.02)
        metrics = calc.calculate(nav_history, trades, benchmark_data)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate(
        self, 
        nav_history: pl.DataFrame,
        trades: List[Trade],
        benchmark_data: Optional[pl.DataFrame] = None
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.
        
        Args:
            nav_history: Daily portfolio values from accounting
            trades: List of executed trades
            benchmark_data: Optional benchmark data (e.g., SPY)
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if nav_history.height == 0:
            return self._empty_metrics()
            
        # Calculate returns and basic statistics
        returns_data = self._calculate_returns(nav_history)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(nav_history, returns_data)
        
        # Calculate risk-adjusted returns
        risk_adjusted = self._calculate_risk_adjusted_returns(
            returns_data, risk_metrics, benchmark_data
        )
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trades, nav_history)
        
        # Calculate efficiency metrics
        efficiency = self._calculate_efficiency_metrics(returns_data, trades)
        
        return PerformanceMetrics(
            # Returns
            total_return=returns_data['total_return'],
            cagr=returns_data['cagr'],
            annualized_return=returns_data['annualized_return'],
            daily_return_mean=returns_data['daily_return_mean'],
            daily_return_std=returns_data['daily_return_std'],
            
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
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            expectancy=trade_stats['expectancy'],
            trade_count=int(trade_stats['trade_count']),
            avg_holding_period=trade_stats['avg_holding_period'],
            
            # Efficiency
            turnover=efficiency['turnover'],
            var_95=efficiency['var_95'],
            skewness=efficiency['skewness'],
            kurtosis=efficiency['kurtosis']
        )
        
    def _calculate_returns(self, nav_history: pl.DataFrame) -> Dict[str, Any]:
        """Calculate return statistics."""
        # Calculate daily returns
        returns_df = nav_history.with_columns([
            (pl.col('total_value') / pl.col('total_value').shift(1) - 1).alias('daily_return')
        ]).fill_null(0)
        
        daily_returns = returns_df.select('daily_return').to_series().to_list()
        
        # Basic statistics
        daily_return_mean = float(np.mean(daily_returns))
        daily_return_std = float(np.std(daily_returns))
        
        # Total return
        initial_value = nav_history.select('total_value').to_series().to_list()[0]
        final_value = nav_history.select('total_value').to_series().to_list()[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time period calculations
        start_date = nav_history.select('timestamp').to_series().to_list()[0]
        end_date = nav_history.select('timestamp').to_series().to_list()[-1]
        
        # Parse dates (assuming YYYY-MM-DD format)
        start_dt = datetime.strptime(start_date.split()[0], '%Y-%m-%d')
        end_dt = datetime.strptime(end_date.split()[0], '%Y-%m-%d')
        years = (end_dt - start_dt).days / 365.25
        
        # CAGR and annualized return
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0
        annualized_return = daily_return_mean * 252  # Trading days per year
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_return': annualized_return,
            'daily_return_mean': daily_return_mean,
            'daily_return_std': daily_return_std,
            'daily_returns': daily_returns
        }
        
    def _calculate_risk_metrics(self, nav_history: pl.DataFrame, returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-related metrics."""
        daily_returns = returns_data['daily_returns']
        
        # Volatility (annualized)
        volatility = returns_data['daily_return_std'] * np.sqrt(252)
        
        # Maximum drawdown
        peak_values = []
        drawdowns = []
        peak = nav_history.select('total_value').to_series().to_list()[0]
        
        for value in nav_history.select('total_value').to_series().to_list():
            if value > peak:
                peak = value
            peak_values.append(peak)
            
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
            
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Maximum drawdown duration
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd > 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        # Calmar ratio
        calmar_ratio = returns_data['cagr'] / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': max_duration,
            'calmar_ratio': float(calmar_ratio)
        }
        
    def _calculate_risk_adjusted_returns(
        self, 
        returns_data: Dict[str, Any], 
        risk_metrics: Dict[str, Any],
        benchmark_data: Optional[pl.DataFrame]
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        daily_returns = returns_data['daily_returns']
        
        # Sharpe ratio
        excess_return = returns_data['annualized_return'] - self.risk_free_rate
        sharpe_ratio = excess_return / risk_metrics['volatility'] if risk_metrics['volatility'] > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0
        downside_volatility = downside_std * np.sqrt(252)
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Information ratio (vs benchmark)
        information_ratio = 0.0
        beta = 0.0
        alpha = 0.0
        
        if benchmark_data is not None and benchmark_data.height > 0:
            # Calculate benchmark returns
            benchmark_returns = benchmark_data.with_columns([
                (pl.col('close') / pl.col('close').shift(1) - 1).alias('bm_return')
            ]).fill_null(0).select('bm_return').to_series().to_list()
            
            # Align returns (use minimum length)
            min_length = min(len(daily_returns), len(benchmark_returns))
            if min_length > 1:
                portfolio_returns = daily_returns[:min_length]
                bm_returns = benchmark_returns[:min_length]
                
                # Calculate tracking error
                tracking_error = np.std([p - b for p, b in zip(portfolio_returns, bm_returns)])
                tracking_error_annual = tracking_error * np.sqrt(252)
                
                # Information ratio
                portfolio_excess_return = np.mean(portfolio_returns) - np.mean(bm_returns)
                information_ratio = portfolio_excess_return / tracking_error if tracking_error > 0 else 0
                
                # Beta and alpha (CAPM)
                if len(bm_returns) > 1 and np.std(bm_returns) > 0:
                    covariance = np.cov(portfolio_returns, bm_returns)[0][1]
                    benchmark_variance = np.var(bm_returns)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    # Alpha (annualized)
                    portfolio_return_annual = np.mean(portfolio_returns) * 252
                    benchmark_return_annual = np.mean(bm_returns) * 252
                    alpha = portfolio_return_annual - (self.risk_free_rate + beta * (benchmark_return_annual - self.risk_free_rate))
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'information_ratio': float(information_ratio),
            'beta': float(beta),
            'alpha': float(alpha)
        }
        
    def _calculate_trade_statistics(self, trades: List[Trade], nav_history: pl.DataFrame) -> Dict[str, Any]:
        """Calculate trade-related statistics."""
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
            
        # Pair buy and sell trades
        buy_trades = [t for t in trades if t.side == 'BUY']
        sell_trades = [t for t in trades if t.side == 'SELL']
        
        # Calculate P&L for each closed position
        closed_trades_pnl = []
        holding_periods = []
        
        position_queue = []  # Queue of open positions (FIFO)
        
        for sell_trade in sell_trades:
            remaining_quantity = sell_trade.quantity
            
            while remaining_quantity > 0 and position_queue:
                buy_trade = position_queue[0]
                
                # Calculate quantity for this pair
                pair_quantity = min(buy_trade.quantity, remaining_quantity)
                
                # Calculate P&L
                proceeds = pair_quantity * sell_trade.price
                cost = pair_quantity * buy_trade.price
                pnl = proceeds - cost - sell_trade.commission
                
                closed_trades_pnl.append(pnl)
                
                # Calculate holding period
                try:
                    buy_date = datetime.strptime(buy_trade.timestamp.split()[0], '%Y-%m-%d')
                    sell_date = datetime.strptime(sell_trade.timestamp.split()[0], '%Y-%m-%d')
                    holding_period = (sell_date - buy_date).days
                    holding_periods.append(holding_period)
                except:
                    holding_periods.append(0)
                
                # Update quantities
                buy_trade.quantity -= pair_quantity
                remaining_quantity -= pair_quantity
                
                if buy_trade.quantity <= 0:
                    position_queue.pop(0)
                    
        # Add remaining buy trades to queue
        for buy_trade in buy_trades:
            if buy_trade.quantity > 0:
                position_queue.append(buy_trade)
        
        # Calculate statistics
        profitable_trades = len([pnl for pnl in closed_trades_pnl if pnl > 0])
        total_trades = len(closed_trades_pnl)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Average win and loss
        wins = [pnl for pnl in closed_trades_pnl if pnl > 0]
        losses = [pnl for pnl in closed_trades_pnl if pnl < 0]
        
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Expectancy
        expectancy = float(np.mean(closed_trades_pnl)) if closed_trades_pnl else 0.0
        
        # Average holding period
        avg_holding_period = float(np.mean(holding_periods)) if holding_periods else 0
        
        return {
            'win_rate': float(win_rate),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': float(profit_factor),
            'expectancy': expectancy,
            'trade_count': len(trades),
            'avg_holding_period': avg_holding_period
        }
        
    def _calculate_efficiency_metrics(self, returns_data: Dict[str, Any], trades: List[Trade]) -> Dict[str, float]:
        """Calculate efficiency and distribution metrics."""
        daily_returns = returns_data['daily_returns']
        
        # Turnover (trading volume relative to portfolio)
        total_volume = sum(trade.quantity * trade.price for trade in trades)
        avg_portfolio_value = np.mean([trade.price * trade.quantity for trade in trades]) if trades else 0
        turnover = total_volume / avg_portfolio_value if avg_portfolio_value > 0 else 0
        
        # Value at Risk (95%)
        var_95 = float(np.percentile(daily_returns, 5)) if daily_returns else 0
        
        # Skewness and kurtosis
        if len(daily_returns) > 2:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            # Skewness
            skewness = float(np.mean([((r - mean_return) / std_return) ** 3 for r in daily_returns])) if std_return > 0 else 0
            
            # Kurtosis
            kurtosis = float(np.mean([((r - mean_return) / std_return) ** 4 for r in daily_returns])) if std_return > 0 else 0
            kurtosis = kurtosis - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
            
        return {
            'turnover': float(turnover),
            'var_95': var_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for empty data."""
        return PerformanceMetrics(
            total_return=0.0, cagr=0.0, annualized_return=0.0, daily_return_mean=0.0, daily_return_std=0.0,
            volatility=0.0, max_drawdown=0.0, max_drawdown_duration=0, calmar_ratio=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, information_ratio=0.0, beta=0.0, alpha=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0, expectancy=0.0,
            trade_count=0, avg_holding_period=0.0,
            turnover=0.0, var_95=0.0, skewness=0.0, kurtosis=0.0
        )
        
    def calculate_with_duckdb(
        self,
        nav_history: pl.DataFrame,
        trades: List[Trade],
        benchmark_data: Optional[pl.DataFrame] = None
    ) -> PerformanceMetrics:
        """Calculate metrics using DuckDB for even faster analytics."""
        # Create DuckDB connection
        conn = duckdb.connect(':memory:')
        
        try:
            # Register DataFrames with DuckDB
            conn.register('nav', nav_history)
            
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
                conn.register('trades', trades_df)
                
            if benchmark_data is not None:
                conn.register('benchmark', benchmark_data)
            
            # Calculate returns using DuckDB
            returns_query = """
            WITH returns AS (
                SELECT 
                    timestamp,
                    total_value,
                    LAG(total_value) OVER (ORDER BY timestamp) as prev_value,
                    (total_value / LAG(total_value) OVER (ORDER BY timestamp) - 1) as daily_return
                FROM nav
                WHERE prev_value IS NOT NULL
            )
            SELECT 
                AVG(daily_return) as daily_return_mean,
                STDDEV(daily_return) as daily_return_std,
                COUNT(*) as trading_days
            FROM returns
            """
            
            returns_result = conn.execute(returns_query).fetchdf()
            
            # Calculate drawdown using DuckDB
            drawdown_query = """
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
                    (peak_value - total_value) / peak_value as drawdown
                FROM peaks
            )
            SELECT 
                MAX(drawdown) as max_drawdown,
                COUNT(*) as total_days
            FROM drawdowns
            """
            
            drawdown_result = conn.execute(drawdown_query).fetchdf()
            
            # For now, fall back to regular calculation
            # DuckDB integration can be expanded in future versions
            return self.calculate(nav_history, trades, benchmark_data)
            
        finally:
            conn.close()