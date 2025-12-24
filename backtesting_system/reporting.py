"""Report generation for backtesting results.

Generates JSON summaries, CSV trade logs, and Plotly charts.
Supports multiple output formats and customizable reports.
"""

import json
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np

from .accounting import Accounting
from .metrics import PerformanceMetrics, MetricsCalculator
from .simulator import Trade

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    PLOTLY_AVAILABLE = False


class BacktestReport:
    """Comprehensive backtest report generator.
    
    Generates JSON summaries, CSV exports, and interactive charts.
    Supports multiple output formats and customizable content.
    
    Parameters:
        output_dir: Directory to save reports (default: 'results')
        
    Example:
        report = BacktestReport(output_dir='results')
        report.generate_full_report(
            accounting=accounting,
            metrics=metrics,
            trades=trades,
            strategy_name='SMA_Crossover',
            symbol='AAPL'
        )
    """
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_full_report(
        self,
        accounting: Accounting,
        metrics: PerformanceMetrics,
        trades: List[Trade],
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        include_charts: bool = True
    ) -> Dict[str, str]:
        """Generate complete backtest report.
        
        Args:
            accounting: Portfolio accounting results
            metrics: Performance metrics
            trades: List of executed trades
            strategy_name: Name of the strategy
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Initial capital
            include_charts: Whether to generate chart files
            
        Returns:
            Dictionary with paths to generated files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{strategy_name}_{symbol}_{timestamp}"
        
        generated_files = {}
        
        # Generate JSON summary
        json_path = self._generate_json_summary(
            accounting, metrics, trades, strategy_name, symbol,
            start_date, end_date, initial_cash, base_filename
        )
        generated_files['json_summary'] = str(json_path)
        
        # Generate CSV trade log
        csv_path = self._generate_csv_trades(trades, base_filename)
        generated_files['csv_trades'] = str(csv_path)
        
        # Generate CSV portfolio history
        portfolio_csv_path = self._generate_csv_portfolio_history(accounting, base_filename)
        generated_files['csv_portfolio'] = str(portfolio_csv_path)
        
        # Generate charts if requested and Plotly is available
        if include_charts and PLOTLY_AVAILABLE:
            chart_files = self._generate_charts(
                accounting, trades, strategy_name, symbol, base_filename
            )
            generated_files.update(chart_files)
        
        return generated_files
        
    def _generate_json_summary(
        self,
        accounting: Accounting,
        metrics: PerformanceMetrics,
        trades: List[Trade],
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        base_filename: str
    ) -> Path:
        """Generate JSON summary report."""
        # Create comprehensive summary
        summary = {
            'metadata': {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_cash': initial_cash,
                'final_portfolio_value': accounting.get_final_portfolio_value(),
                'total_return': accounting.get_total_return(),
                'trade_count': len(trades),
                'generated_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            
            'performance_metrics': {
                'returns': {
                    'total_return': round(metrics.total_return, 4),
                    'cagr': round(metrics.cagr, 4),
                    'annualized_return': round(metrics.annualized_return, 4),
                    'daily_return_mean': round(metrics.daily_return_mean, 6),
                    'daily_return_std': round(metrics.daily_return_std, 6)
                },
                
                'risk_metrics': {
                    'volatility': round(metrics.volatility, 4),
                    'max_drawdown': round(metrics.max_drawdown, 4),
                    'max_drawdown_duration': metrics.max_drawdown_duration,
                    'calmar_ratio': round(metrics.calmar_ratio, 4),
                    'var_95': round(metrics.var_95, 6)
                },
                
                'risk_adjusted_returns': {
                    'sharpe_ratio': round(metrics.sharpe_ratio, 4),
                    'sortino_ratio': round(metrics.sortino_ratio, 4),
                    'information_ratio': round(metrics.information_ratio, 4),
                    'beta': round(metrics.beta, 4),
                    'alpha': round(metrics.alpha, 4)
                },
                
                'trade_statistics': {
                    'win_rate': round(metrics.win_rate, 4),
                    'avg_win': round(metrics.avg_win, 2),
                    'avg_loss': round(metrics.avg_loss, 2),
                    'profit_factor': round(metrics.profit_factor, 4),
                    'expectancy': round(metrics.expectancy, 2),
                    'trade_count': metrics.trade_count,
                    'avg_holding_period': round(metrics.avg_holding_period, 1)
                },
                
                'efficiency_metrics': {
                    'turnover': round(metrics.turnover, 4),
                    'skewness': round(metrics.skewness, 4),
                    'kurtosis': round(metrics.kurtosis, 4)
                }
            },
            
            'portfolio_breakdown': {
                'initial_cash': initial_cash,
                'final_cash': accounting.cash,
                'final_position': accounting.position,
                'realized_pnl': round(accounting.realized_pnl, 2),
                'unrealized_pnl': round(accounting.unrealized_pnl, 2),
                'total_pnl': round(accounting.get_total_pnl(), 2)
            },
            
            'trade_summary': self._summarize_trades(trades)
        }
        
        # Save JSON file
        json_path = self.output_dir / f"{base_filename}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return json_path
        
    def _generate_csv_trades(self, trades: List[Trade], base_filename: str) -> Path:
        """Generate CSV trade log."""
        csv_path = self.output_dir / f"{base_filename}_trades.csv"
        
        if not trades:
            # Create empty CSV with headers
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'side', 'price', 'quantity', 'commission', 'slippage', 'total_cost'])
            return csv_path
        
        # Write trades to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['timestamp', 'side', 'price', 'quantity', 'commission', 'slippage', 'total_cost'])
            
            # Trade rows
            for trade in trades:
                writer.writerow([
                    trade.timestamp,
                    trade.side,
                    round(trade.price, 4),
                    trade.quantity,
                    round(trade.commission, 4),
                    round(trade.slippage, 4),
                    round(trade.total_cost, 4)
                ])
                
        return csv_path
        
    def _generate_csv_portfolio_history(self, accounting: Accounting, base_filename: str) -> Path:
        """Generate CSV portfolio history."""
        csv_path = self.output_dir / f"{base_filename}_portfolio.csv"
        
        nav_history = accounting.get_nav_history()
        
        if nav_history.height == 0:
            # Create empty CSV with headers
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'cash', 'position_value', 'total_value', 'position_quantity', 'unrealized_pnl', 'realized_pnl', 'total_pnl'])
            return csv_path
        
        # Write portfolio history to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['timestamp', 'cash', 'position_value', 'total_value', 'position_quantity', 'unrealized_pnl', 'realized_pnl', 'total_pnl'])
            
            # Data rows
            for row in nav_history.iter_rows(named=True):
                writer.writerow([
                    row['timestamp'],
                    round(row['cash'], 2),
                    round(row['position_value'], 2),
                    round(row['total_value'], 2),
                    row['position_quantity'],
                    round(row['unrealized_pnl'], 2),
                    round(row['realized_pnl'], 2),
                    round(row['total_pnl'], 2)
                ])
                
        return csv_path
        
    def _generate_charts(
        self,
        accounting: Accounting,
        trades: List[Trade],
        strategy_name: str,
        symbol: str,
        base_filename: str
    ) -> Dict[str, str]:
        """Generate interactive charts using Plotly."""
        if not PLOTLY_AVAILABLE:
            return {}
            
        chart_files = {}
        
        # Get data
        nav_history = accounting.get_nav_history()
        
        if nav_history.height == 0:
            return chart_files
        
        # 1. Equity Curve Chart
        equity_chart_path = self.output_dir / f"{base_filename}_equity_curve.html"
        self._create_equity_curve_chart(nav_history, trades, strategy_name, symbol, equity_chart_path)
        chart_files['equity_curve'] = str(equity_chart_path)
        
        # 2. Drawdown Chart
        drawdown_chart_path = self.output_dir / f"{base_filename}_drawdown.html"
        self._create_drawdown_chart(nav_history, strategy_name, symbol, drawdown_chart_path)
        chart_files['drawdown'] = str(drawdown_chart_path)
        
        # 3. Returns Distribution Chart
        returns_chart_path = self.output_dir / f"{base_filename}_returns.html"
        self._create_returns_distribution_chart(nav_history, strategy_name, symbol, returns_chart_path)
        chart_files['returns_distribution'] = str(returns_chart_path)
        
        # 4. Trade Analysis Chart
        if trades:
            trade_chart_path = self.output_dir / f"{base_filename}_trades.html"
            self._create_trade_analysis_chart(nav_history, trades, strategy_name, symbol, trade_chart_path)
            chart_files['trade_analysis'] = str(trade_chart_path)
        
        return chart_files
        
    def _create_equity_curve_chart(self, nav_history: pl.DataFrame, trades: List[Trade], strategy_name: str, symbol: str, output_path: Path) -> None:
        """Create equity curve chart with trade markers."""
        if not PLOTLY_AVAILABLE or go is None:
            return
            
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=nav_history['timestamp'],
            y=nav_history['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Add cash line
        fig.add_trace(go.Scatter(
            x=nav_history['timestamp'],
            y=nav_history['cash'],
            mode='lines',
            name='Cash',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        # Add trade markers if available
        if trades:
            trade_timestamps = []
            trade_values = []
            trade_colors = []
            trade_text = []
            
            # Get portfolio value at each trade timestamp
            nav_dict = {
                str(ts): val for ts, val in 
                zip(nav_history['timestamp'], nav_history['total_value'])
            }
            
            for trade in trades:
                if trade.timestamp in nav_dict:
                    trade_timestamps.append(trade.timestamp)
                    trade_values.append(nav_dict[trade.timestamp])
                    trade_colors.append('red' if trade.side == 'SELL' else 'green')
                    trade_text.append(f"{trade.side}: {trade.quantity} @ ${trade.price:.2f}")
            
            if trade_timestamps:
                fig.add_trace(go.Scatter(
                    x=trade_timestamps,
                    y=trade_values,
                    mode='markers',
                    name='Trades',
                    marker=dict(color=trade_colors, size=8, symbol='triangle-up'),
                    text=trade_text,
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'Equity Curve: {strategy_name} on {symbol}',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        fig.write_html(str(output_path))
        
    def _create_drawdown_chart(self, nav_history: pl.DataFrame, strategy_name: str, symbol: str, output_path: Path) -> None:
        """Create drawdown chart."""
        if not PLOTLY_AVAILABLE or go is None:
            return
            
        # Calculate drawdown
        peak_values = []
        drawdowns = []
        peak = nav_history.select('total_value').to_series().to_list()[0]
        
        for value in nav_history.select('total_value').to_series().to_list():
            if value > peak:
                peak = value
            peak_values.append(peak)
            
            drawdown = (peak - value) / peak * 100  # Convert to percentage
            drawdowns.append(drawdown)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=nav_history['timestamp'],
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'Drawdown: {strategy_name} on {symbol}',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        fig.write_html(str(output_path))
        
    def _create_returns_distribution_chart(self, nav_history: pl.DataFrame, strategy_name: str, symbol: str, output_path: Path) -> None:
        """Create returns distribution chart."""
        if not PLOTLY_AVAILABLE or go is None or make_subplots is None:
            return
            
        # Calculate daily returns
        returns_df = nav_history.with_columns([
            (pl.col('total_value') / pl.col('total_value').shift(1) - 1).alias('daily_return')
        ]).fill_null(0)
        
        daily_returns = returns_df.select('daily_return').to_series().to_list()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Returns Over Time', 'Returns Distribution'),
            vertical_spacing=0.1
        )
        
        # Time series of returns
        fig.add_trace(
            go.Scatter(
                x=returns_df['timestamp'],
                y=returns_df['daily_return'],
                mode='lines',
                name='Daily Return',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Histogram of returns
        fig.add_trace(
            go.Histogram(
                x=daily_returns,
                nbinsx=50,
                name='Frequency',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Returns Analysis: {strategy_name} on {symbol}',
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        fig.update_yaxes(title_text='Daily Return', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Daily Return', row=2, col=1)
        
        fig.write_html(str(output_path))
        
    def _create_trade_analysis_chart(self, nav_history: pl.DataFrame, trades: List[Trade], strategy_name: str, symbol: str, output_path: Path) -> None:
        """Create trade analysis chart."""
        if not trades or not PLOTLY_AVAILABLE or go is None or make_subplots is None:
            return
            
        # Calculate trade P&L
        trade_pnl = []
        trade_types = []
        
        position_queue = []
        
        for trade in trades:
            if trade.side == 'BUY':
                position_queue.append(trade)
            elif trade.side == 'SELL' and position_queue:
                # Calculate P&L for FIFO trades
                remaining_qty = trade.quantity
                total_pnl = 0
                
                while remaining_qty > 0 and position_queue:
                    buy_trade = position_queue[0]
                    qty = min(buy_trade.quantity, remaining_qty)
                    
                    proceeds = qty * trade.price
                    cost = qty * buy_trade.price
                    pnl = proceeds - cost - trade.commission * (qty / trade.quantity)
                    
                    total_pnl += pnl
                    remaining_qty -= qty
                    buy_trade.quantity -= qty
                    
                    if buy_trade.quantity <= 0:
                        position_queue.pop(0)
                
                trade_pnl.append(total_pnl)
                trade_types.append('SELL')
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trade P&L', 'Cumulative P&L'),
            vertical_spacing=0.1
        )
        
        # Individual trade P&L
        colors = ['green' if pnl >= 0 else 'red' for pnl in trade_pnl]
        fig.add_trace(
            go.Bar(
                x=list(range(len(trade_pnl))),
                y=trade_pnl,
                name='Trade P&L',
                marker_color=colors,
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(trade_pnl)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_pnl))),
                y=cumulative_pnl,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Trade Analysis: {strategy_name} on {symbol}',
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        fig.update_yaxes(title_text='P&L ($)', row=1, col=1)
        fig.update_yaxes(title_text='Cumulative P&L ($)', row=2, col=1)
        fig.update_xaxes(title_text='Trade Number', row=1, col=1)
        fig.update_xaxes(title_text='Trade Number', row=2, col=1)
        
        fig.write_html(str(output_path))
        
    def _summarize_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Create summary statistics for trades."""
        if not trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_commission': 0,
                'total_slippage': 0,
                'avg_trade_size': 0
            }
        
        buy_trades = [t for t in trades if t.side == 'BUY']
        sell_trades = [t for t in trades if t.side == 'SELL']
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': round(sum(t.commission for t in trades), 2),
            'total_slippage': round(sum(t.slippage for t in trades), 2),
            'avg_trade_size': round(np.mean([t.quantity for t in trades]), 0),
            'total_volume': round(sum(t.quantity * t.price for t in trades), 2)
        }