"""
Streamlit Dashboard for Stock Backtesting System

Interactive local dashboard with 4 pages:
1. Strategy Runner - Configure and run backtests
2. Results Dashboard - View performance metrics and charts
3. Strategy Comparison - Compare multiple strategies side-by-side
4. Trade Log - Detailed trade analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import yaml

from backtesting_system import (
    BacktestConfig, run_backtest_with_report, run_multiple_backtests,
    DataLoader, MetricsCalculator, Accounting
)
from backtesting_system.validation import BiasValidator
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy


# Strategy registry
STRATEGIES = {
    'SMA Crossover': SMAStrategy,
    'Momentum': MomentumStrategy,
    'Mean Reversion': MeanReversionStrategy,
}

# Default symbols
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'VTI']


def set_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Stock Backtesting System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def load_strategy_config() -> Dict[str, Any]:
    """Load strategy configuration from file"""
    try:
        with open('config/strategies.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}


def save_backtest_to_session(result: Dict[str, Any]):
    """Save backtest result to session state"""
    if 'backtests' not in st.session_state:
        st.session_state.backtests = []
    
    # Add timestamp
    result['run_time'] = datetime.now().isoformat()
    st.session_state.backtests.append(result)
    
    # Keep only last 10 results
    if len(st.session_state.backtests) > 10:
        st.session_state.backtests = st.session_state.backtests[-10:]


def get_strategy_params_ui(strategy_name: str) -> Dict[str, Any]:
    """Get strategy parameters from UI"""
    params = {}
    
    if strategy_name == 'SMA Crossover':
        col1, col2 = st.columns(2)
        with col1:
            params['fast_period'] = st.slider('Fast MA Period', 5, 50, 10)
        with col2:
            params['slow_period'] = st.slider('Slow MA Period', 20, 200, 50)
    
    elif strategy_name == 'Momentum':
        col1, col2 = st.columns(2)
        with col1:
            params['lookback_period'] = st.slider('Lookback Period', 5, 100, 20)
        with col2:
            params['momentum_threshold'] = st.slider(
                'Momentum Threshold', 0.01, 0.1, 0.02, step=0.01, format="%.3f"
            )
    
    elif strategy_name == 'Mean Reversion':
        col1, col2 = st.columns(2)
        with col1:
            params['ma_period'] = st.slider('MA Period', 10, 50, 20)
        with col2:
            params['threshold'] = st.slider(
                'Distance Threshold', 0.01, 0.2, 0.05, step=0.01, format="%.3f"
            )
    
    return params


def plot_equity_curve(nav_history: List[Dict[str, Any]], title: str = "Portfolio Equity Curve"):
    """Plot equity curve"""
    if not nav_history:
        return go.Figure()
    
    df = pd.DataFrame(nav_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['nav'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_drawdown(nav_history: List[Dict[str, Any]], title: str = "Drawdown"):
    """Plot drawdown chart"""
    if not nav_history:
        return go.Figure()
    
    df = pd.DataFrame(nav_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['nav'] = pd.to_numeric(df['nav'])
    
    # Calculate drawdown
    df['peak'] = df['nav'].expanding().max()
    df['drawdown'] = (df['nav'] - df['peak']) / df['peak']
    
    fig = go.Figure()
    
    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['drawdown'] * 100,
        mode='lines',
        name='Drawdown',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_trades_on_chart(data_df: pd.DataFrame, trades: List[Dict[str, Any]], title: str = "Price Chart with Trades"):
    """Plot price chart with trade markers"""
    if data_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(data_df['timestamp']),
        y=data_df['close'],
        mode='lines',
        name='Price',
        line=dict(color='black', width=1)
    ))
    
    # Add trade markers
    if trades:
        buys = [t for t in trades if t['side'] == 'BUY']
        sells = [t for t in trades if t['side'] == 'SELL']
        
        if buys:
            buy_times = pd.to_datetime([t['timestamp'] for t in buys])
            buy_prices = [t['price'] for t in buys]
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        if sells:
            sell_times = pd.to_datetime([t['timestamp'] for t in sells])
            sell_prices = [t['price'] for t in sells]
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def page_strategy_runner():
    """Page 1: Strategy Runner"""
    st.header("🚀 Strategy Runner")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Strategy Configuration")
        
        # Strategy selection
        strategy_name = st.selectbox("Select Strategy", list(STRATEGIES.keys()))
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        strategy_params = get_strategy_params_ui(strategy_name)
        
        # Symbol and date range
        st.subheader("Data Configuration")
        
        symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS + ["Custom..."])
        if symbol == "Custom...":
            symbol = st.text_input("Enter Symbol", "AAPL").upper()
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))
        with col_date2:
            end_date = st.date_input("End Date", datetime.now() - timedelta(days=1))
    
    with col2:
        st.subheader("Simulation Settings")
        
        initial_cash = st.number_input(
            "Initial Cash ($)", 10000, 1000000, 100000, step=10000
        )
        
        commission_bps = st.slider(
            "Commission (bps)", 0, 20, 5, help="Commission in basis points (0.01%)"
        )
        
        slippage_pct = st.slider(
            "Slippage (%)", 0.0, 0.5, 0.1, step=0.05, format="%.2f"
        )
        
        position_size_pct = st.slider(
            "Position Size (%)", 10, 100, 100, step=10
        )
        
        # Bias validation
        enable_validation = st.checkbox("Enable Bias Validation", value=True)
    
    # Run button
    st.markdown("---")
    col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
    with col_run2:
        run_button = st.button("🎯 RUN BACKTEST", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner("Running backtest..."):
            try:
                # Create configuration
                config = BacktestConfig(
                    strategy_class=STRATEGIES[strategy_name],
                    strategy_params=strategy_params,
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    initial_cash=initial_cash,
                    commission_bps=commission_bps,
                    slippage_pct=slippage_pct,
                    position_size_pct=position_size_pct,
                )
                
                # Run backtest
                result = run_backtest_with_report(config)
                
                if result['success']:
                    # Save to session
                    save_backtest_to_session(result)
                    
                    # Success message
                    st.success("✅ Backtest completed successfully!")
                    
                    # Key metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("Final Portfolio", f"${result['final_portfolio_value']:,.2f}")
                    with col_metrics2:
                        st.metric("Total Return", f"{result['total_return']:.2%}")
                    with col_metrics3:
                        st.metric("Number of Trades", result['num_trades'])
                    
                    # Bias validation
                    if enable_validation and result['trades']:
                        with st.expander("🔍 Bias Validation Results", expanded=True):
                            # Get data for validation
                            loader = DataLoader()
                            data = loader.load(symbol, config.start_date, config.end_date)
                            strategy = STRATEGIES[strategy_name](**strategy_params)
                            signals = strategy.generate_signals(data)
                            
                            validator = BiasValidator()
                            
                            # Calculate buy-and-hold returns
                            buy_hold_returns = data.select('close').to_series().pct_change().to_list()
                            
                            validation_results = validator.validate_all(
                                data.to_pandas(),
                                signals.to_pandas(),
                                result['trades'],
                                commission_bps,
                                slippage_pct,
                                buy_hold_returns=buy_hold_returns
                            )
                            
                            validator.print_results()
                    
                else:
                    st.error(f"❌ Backtest failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error running backtest: {str(e)}")


def page_results_dashboard():
    """Page 2: Results Dashboard"""
    st.header("📊 Results Dashboard")
    
    if 'backtests' not in st.session_state or not st.session_state.backtests:
        st.info("No backtest results available. Run a backtest first on the Strategy Runner page.")
        return
    
    # Select backtest to view
    backtest_options = [
        f"{bt['config']['strategy_name']} on {bt['config']['symbol']} "
        f"({bt['config']['start_date']} to {bt['config']['end_date']})"
        for bt in st.session_state.backtests
    ]
    
    selected_index = st.selectbox("Select Backtest", range(len(backtest_options)), 
                                  format_func=lambda i: backtest_options[i])
    result = st.session_state.backtests[selected_index]
    
    if not result['success']:
        st.error(f"❌ Backtest failed: {result['error']}")
        return
    
    # Key metrics
    st.subheader("📈 Performance Summary")
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    
    with col_metrics1:
        st.metric("Final Portfolio", f"${result['final_portfolio_value']:,.2f}")
    with col_metrics2:
        st.metric("Total Return", f"{result['total_return']:.2%}")
    with col_metrics3:
        st.metric("Number of Trades", result['num_trades'])
    with col_metrics4:
        st.metric("Execution Time", f"{result['execution_time_seconds']:.2f}s")
    
    # Charts
    st.subheader("📊 Performance Charts")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if result['nav_history']:
            st.plotly_chart(plot_equity_curve(result['nav_history']), use_container_width=True)
    
    with col_chart2:
        if result['nav_history']:
            st.plotly_chart(plot_drawdown(result['nav_history']), use_container_width=True)
    
    # Detailed metrics
    if result['metrics']:
        st.subheader("📋 Detailed Metrics")
        
        metrics = result['metrics']
        
        # Return metrics
        col_return1, col_return2, col_return3 = st.columns(3)
        with col_return1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 'N/A'):.2f}")
        with col_return2:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 'N/A'):.2f}")
        with col_return3:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 'N/A'):.2f}")
        
        # Risk metrics
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        with col_risk1:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        with col_risk2:
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
        with col_risk3:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
    
    # Price chart with trades
    if result['trades']:
        st.subheader("💰 Price Chart with Trades")
        
        try:
            loader = DataLoader()
            data = loader.load(
                result['config']['symbol'],
                result['config']['start_date'],
                result['config']['end_date']
            )
            st.plotly_chart(
                plot_trades_on_chart(data.to_pandas(), result['trades']), 
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Could not load price data: {e}")
    
    # Export options
    st.subheader("💾 Export Results")
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("Export JSON"):
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"backtest_{result['config']['symbol']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col_export2:
        if result['trades'] and st.button("Export Trade Log"):
            trades_df = pd.DataFrame(result['trades'])
            csv_data = trades_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"trades_{result['config']['symbol']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def page_strategy_comparison():
    """Page 3: Strategy Comparison"""
    st.header("⚖️ Strategy Comparison")
    
    if 'backtests' not in st.session_state or len(st.session_state.backtests) < 2:
        st.info("Need at least 2 backtest results for comparison. Run more backtests first.")
        return
    
    # Select backtests to compare
    backtest_options = [
        f"{bt['config']['strategy_name']} on {bt['config']['symbol']}"
        for bt in st.session_state.backtests
    ]
    
    selected_indices = st.multiselect(
        "Select Backtests to Compare (max 5)", 
        range(len(backtest_options)),
        format_func=lambda i: backtest_options[i],
        max_selections=5
    )
    
    if not selected_indices:
        return
    
    # Get selected results
    selected_results = [st.session_state.backtests[i] for i in selected_indices]
    
    # Filter successful results
    successful_results = [r for r in selected_results if r['success']]
    
    if not successful_results:
        st.error("No successful backtests selected for comparison.")
        return
    
    # Comparison table
    st.subheader("📊 Performance Comparison")
    
    comparison_data = []
    for result in successful_results:
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': result['config']['strategy_name'],
            'Symbol': result['config']['symbol'],
            'Total Return': f"{result['total_return']:.2%}",
            'Final Value': f"${result['final_portfolio_value']:,.0f}",
            'Trades': result['num_trades'],
            'Sharpe': f"{metrics.get('sharpe_ratio', 'N/A'):.2f}" if metrics.get('sharpe_ratio') else 'N/A',
            'Max DD': f"{metrics.get('max_drawdown', 0):.2%}",
            'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Equity curves comparison
    if len(successful_results) > 1:
        st.subheader("📈 Equity Curves Comparison")
        
        fig = go.Figure()
        
        for result in successful_results:
            if result['nav_history']:
                df = pd.DataFrame(result['nav_history'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                label = f"{result['config']['strategy_name']} ({result['config']['symbol']})"
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['nav'],
                    mode='lines',
                    name=label,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Portfolio Value Comparison",
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk/Return scatter
    if len(successful_results) > 2:
        st.subheader("📊 Risk vs Return Analysis")
        
        scatter_data = []
        for result in successful_results:
            metrics = result['metrics']
            if metrics.get('sharpe_ratio') and metrics.get('max_drawdown'):
                scatter_data.append({
                    'Strategy': result['config']['strategy_name'],
                    'Symbol': result['config']['symbol'],
                    'Return': result['total_return'],
                    'Sharpe': metrics.get('sharpe_ratio'),
                    'Max Drawdown': abs(metrics.get('max_drawdown', 0)),
                })
        
        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=scatter_df['Max Drawdown'],
                y=scatter_df['Return'],
                mode='markers+text',
                text=scatter_df.apply(lambda x: f"{x['Strategy']}<br>{x['Symbol']}", axis=1),
                textposition='top center',
                marker=dict(
                    size=scatter_df['Sharpe'] * 50,  # Size by Sharpe ratio
                    color=scatter_df['Sharpe'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Strategies'
            ))
            
            fig.update_layout(
                title="Risk vs Return (Bubble size = Sharpe Ratio)",
                xaxis_title='Max Drawdown (Risk)',
                yaxis_title='Total Return',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)


def page_trade_log():
    """Page 4: Trade Log"""
    st.header("📋 Trade Log Analysis")
    
    if 'backtests' not in st.session_state or not st.session_state.backtests:
        st.info("No backtest results available. Run a backtest first.")
        return
    
    # Select backtest
    backtest_options = [
        f"{bt['config']['strategy_name']} on {bt['config']['symbol']} "
        f"({bt['config']['start_date']} to {bt['config']['end_date']})"
        for bt in st.session_state.backtests if bt['success']
    ]
    
    if not backtest_options:
        st.error("No successful backtests available.")
        return
    
    selected_index = st.selectbox("Select Backtest", range(len(backtest_options)), 
                                  format_func=lambda i: backtest_options[i])
    
    successful_backtests = [bt for bt in st.session_state.backtests if bt['success']]
    result = successful_backtests[selected_index]
    
    if not result['trades']:
        st.warning("No trades executed in this backtest.")
        return
    
    # Trade summary
    st.subheader("📊 Trade Summary")
    
    trades = result['trades']
    buys = [t for t in trades if t['side'] == 'BUY']
    sells = [t for t in trades if t['side'] == 'SELL']
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("Total Trades", len(trades))
        st.metric("Buy Trades", len(buys))
    with col_summary2:
        st.metric("Sell Trades", len(sells))
        if result['metrics'].get('winning_trades') and result['metrics'].get('losing_trades'):
            st.metric("Winning Trades", result['metrics']['winning_trades'])
    with col_summary3:
        if result['metrics'].get('win_rate'):
            st.metric("Win Rate", f"{result['metrics']['win_rate']:.2%}")
        if result['metrics'].get('avg_win'):
            st.metric("Avg Win", f"${result['metrics']['avg_win']:.2f}")
    
    # Trade table
    st.subheader("📋 Detailed Trade Log")
    
    trades_df = pd.DataFrame(trades)
    
    # Format columns
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d')
    if 'price' in trades_df.columns:
        trades_df['price'] = trades_df['price'].round(2)
    if 'total_value' in trades_df.columns:
        trades_df['total_value'] = trades_df['total_value'].round(2)
    if 'commission' in trades_df.columns:
        trades_df['commission'] = trades_df['commission'].round(2)
    if 'slippage' in trades_df.columns:
        trades_df['slippage'] = trades_df['slippage'].round(2)
    
    # Display table
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    # Trade analysis
    if len(trades) > 1:
        st.subheader("📈 Trade Analysis")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            # Trade frequency
            if 'timestamp' in trades_df.columns:
                trades_df['date'] = pd.to_datetime(trades_df['timestamp'])
                trades_df['month'] = trades_df['date'].dt.to_period('M')
                trades_per_month = trades_df.groupby('month').size()
                
                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Bar(
                    x=trades_per_month.index.astype(str),
                    y=trades_per_month.values,
                    name='Trades per Month'
                ))
                fig_monthly.update_layout(
                    title="Trade Frequency (Monthly)",
                    xaxis_title='Month',
                    yaxis_title='Number of Trades',
                    template='plotly_white'
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col_analysis2:
            # Trade P&L distribution
            if 'pnl' in trades_df.columns:
                pnl_data = trades_df['pnl'].dropna()
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Histogram(
                    x=pnl_data,
                    nbinsx=20,
                    name='P&L Distribution'
                ))
                fig_pnl.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title='P&L ($)',
                    yaxis_title='Count',
                    template='plotly_white'
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Trade Log")
    
    csv_data = trades_df.to_csv(index=False)
    st.download_button(
        label="Download Trade Log (CSV)",
        data=csv_data,
        file_name=f"trades_{result['config']['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def main():
    """Main Streamlit app"""
    set_page_config()
    
    # Sidebar navigation
    st.sidebar.title("📊 Stock Backtesting System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🚀 Strategy Runner",
            "📊 Results Dashboard", 
            "⚖️ Strategy Comparison",
            "📋 Trade Log"
        ]
    )
    
    # Session state initialization
    if 'backtests' not in st.session_state:
        st.session_state.backtests = []
    
    # Render selected page
    if page == "🚀 Strategy Runner":
        page_strategy_runner()
    elif page == "📊 Results Dashboard":
        page_results_dashboard()
    elif page == "⚖️ Strategy Comparison":
        page_strategy_comparison()
    elif page == "📋 Trade Log":
        page_trade_log()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Stock Backtesting System V1**
        
        Built with:
        - 🐍 Python & Streamlit
        - 📊 Polars & DuckDB  
        - ⚡ Vectorized Execution
        """
    )


if __name__ == "__main__":
    main()