"""Week 3 Integration Demo

Demonstrates the new accounting, metrics, and reporting functionality.
Shows complete end-to-end backtesting with Week 3 features.
"""

import polars as pl
import numpy as np
from pathlib import Path

# Import Week 3 components
from backtesting_system.data_loader import DataLoader
from backtesting_system.strategy import BaseStrategy
from strategies.sma_strategy import SMAStrategy
from backtesting_system.simulator import Simulator
from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator
from backtesting_system.reporting import BacktestReport


def demo_week3_features():
    """Demonstrate Week 3 accounting, metrics, and reporting features."""
    
    print("🚀 Week 3 Backtesting System Demo")
    print("=" * 50)
    
    # 1. Load data
    print("\n📊 1. Loading Historical Data...")
    loader = DataLoader(cache_dir="data/raw")
    
    try:
        df = loader.download("AAPL", start="2020-01-01", end="2024-12-31")
        print(f"   ✓ Loaded {df.height} days of AAPL data")
    except Exception as e:
        print(f"   ⚠ Using sample data due to: {e}")
        # Create sample data if download fails
        dates = [f"2020-{i//30+1:02d}-{i%30+1:02d}" for i in range(365)]
        prices = [100 + i * 0.1 + np.sin(i * 0.1) * 5 for i in range(365)]
        
        df = pl.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        })
        print(f"   ✓ Created {df.height} days of sample data")
    
    # 2. Generate strategy signals
    print("\n🎯 2. Generating Strategy Signals...")
    strategy = SMAStrategy(fast_period=10, slow_period=50)
    df_with_signals = strategy.generate_signals(df)
    print(f"   ✓ Generated SMA crossover signals")
    
    # Show signal distribution
    signal_counts = df_with_signals['signal'].value_counts()
    print(f"   Signal Distribution: {signal_counts.to_dicts()}")
    
    # 3. Execute trades with simulator
    print("\n⚡ 3. Executing Trades...")
    simulator = Simulator(
        initial_cash=100000,
        commission_bps=5,      # 5 bps = 0.05%
        slippage_pct=0.001,   # 0.1% slippage
        position_size_pct=1.0   # Use 100% of cash
    )
    
    sim_result = simulator.execute(df_with_signals)
    print(f"   ✓ Executed {sim_result['num_trades']} trades")
    print(f"   ✓ Final Portfolio Value: ${sim_result['final_portfolio_value']:,.2f}")
    print(f"   ✓ Total Return: {sim_result['total_return']:.2%}")
    print(f"   ✓ Total Commission: ${sim_result.get('total_commission', 0):.2f}")
    print(f"   ✓ Total Slippage: ${sim_result.get('total_slippage', 0):.2f}")
    
    # 4. Process with Accounting (NEW WEEK 3)
    print("\n📈 4. Portfolio Accounting...")
    accounting = Accounting(initial_cash=100000)
    accounting.process_trades(sim_result['trades'], df_with_signals)
    
    print(f"   ✓ Final Cash: ${accounting.cash:,.2f}")
    print(f"   ✓ Final Position: {accounting.position} shares")
    print(f"   ✓ Realized P&L: ${accounting.realized_pnl:,.2f}")
    print(f"   ✓ Unrealized P&L: ${accounting.unrealized_pnl:,.2f}")
    print(f"   ✓ Total P&L: ${accounting.get_total_pnl():,.2f}")
    print(f"   ✓ Win Rate: {accounting.get_win_rate():.1%}")
    
    # 5. Calculate Performance Metrics (NEW WEEK 3)
    print("\n📊 5. Performance Metrics...")
    calculator = MetricsCalculator(risk_free_rate=0.02)
    nav_history = accounting.get_nav_history()
    metrics = calculator.calculate(nav_history, sim_result['trades'])
    
    print(f"   ✓ CAGR: {metrics.cagr:.2%}")
    print(f"   ✓ Volatility: {metrics.volatility:.2%}")
    print(f"   ✓ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   ✓ Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"   ✓ Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   ✓ Max DD Duration: {metrics.max_drawdown_duration} days")
    print(f"   ✓ Win Rate: {metrics.win_rate:.1%}")
    print(f"   ✓ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   ✓ Trade Count: {metrics.trade_count}")
    
    # 6. Generate Reports (NEW WEEK 3)
    print("\n📄 6. Generating Reports...")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    report = BacktestReport(output_dir=str(results_dir))
    
    # Generate full report
    report_files = report.generate_full_report(
        accounting=accounting,
        metrics=metrics,
        trades=sim_result['trades'],
        strategy_name='SMA_Crossover',
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_cash=100000,
        include_charts=True
    )
    
    print("   ✓ Generated report files:")
    for file_type, file_path in report_files.items():
        file_size = Path(file_path).stat().st_size
        print(f"     - {file_type}: {Path(file_path).name} ({file_size:,} bytes)")
    
    # 7. Show sample trade log
    print("\n📋 7. Sample Trade Log:")
    if sim_result['trades']:
        print("   Date        | Side | Price   | Qty | Commission | Slippage | Total")
        print("   ------------|------|---------|-----|------------|-----------|-------")
        
        for i, trade in enumerate(sim_result['trades'][:5]):  # Show first 5 trades
            print(f"   {trade.timestamp[:10]} | {trade.side:4} | {trade.price:7.2f} | {trade.quantity:3} | {trade.commission:10.2f} | {trade.slippage:9.2f} | {trade.total_cost:6.2f}")
        
        if len(sim_result['trades']) > 5:
            print(f"   ... and {len(sim_result['trades']) - 5} more trades")
    
    # 8. Performance Summary
    print("\n🎯 8. Performance Summary:")
    print("   " + "="*60)
    print("   " + f"Strategy: SMA Crossover (10/50)")
    print("   " + f"Symbol:   AAPL")
    print("   " + f"Period:   2020-2024")
    print("   " + "-"*60)
    print("   " + f"Initial Capital:     ${100000:,.2f}")
    print("   " + f"Final Portfolio:    ${metrics.cagr * 100000 + 100000:,.2f}")
    print("   " + f"Total Return:        {accounting.get_total_return():.2%}")
    print("   " + f"Annual Return:       {metrics.cagr:.2%}")
    print("   " + f"Volatility:          {metrics.volatility:.2%}")
    print("   " + f"Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
    print("   " + f"Max Drawdown:        {metrics.max_drawdown:.2%}")
    print("   " + f"Win Rate:            {accounting.get_win_rate():.1%}")
    print("   " + f"Number of Trades:    {len(sim_result['trades'])}")
    print("   " + "="*60)
    
    print(f"\n✅ Week 3 Demo Complete! Check {results_dir} for detailed reports.")
    
    return {
        'accounting': accounting,
        'metrics': metrics,
        'trades': sim_result['trades'],
        'report_files': report_files
    }


def compare_strategies_demo():
    """Demonstrate comparing multiple strategies."""
    
    print("\n🏁 Strategy Comparison Demo")
    print("=" * 50)
    
    # Create sample data
    dates = [f"2020-{i//30+1:02d}-{i%30+1:02d}" for i in range(180)]  # 6 months
    prices = [100 + i * 0.05 + np.sin(i * 0.05) * 2 for i in range(180)]
    
    df = pl.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    })
    
    strategies = [
        ('SMA_10_30', SMAStrategy(fast_period=10, slow_period=30)),
        ('SMA_20_50', SMAStrategy(fast_period=20, slow_period=50)),
        ('SMA_5_15', SMAStrategy(fast_period=5, slow_period=15))
    ]
    
    results = {}
    calculator = MetricsCalculator()
    
    for name, strategy in strategies:
        print(f"\n📊 Testing {name}...")
        
        # Generate signals
        df_signals = strategy.generate_signals(df)
        
        # Execute
        sim = Simulator(initial_cash=100000)
        sim_result = sim.execute(df_signals)
        
        # Account
        acct = Accounting(initial_cash=100000)
        acct.process_trades(sim_result['trades'], df_signals)
        
        # Metrics
        nav_history = acct.get_nav_history()
        metrics = calculator.calculate(nav_history, sim_result['trades'])
        
        results[name] = {
            'total_return': acct.get_total_return(),
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'trade_count': len(sim_result['trades']),
            'win_rate': acct.get_win_rate()
        }
    
    # Display comparison
    print("\n📈 Strategy Comparison:")
    print("Strategy     | Return | Sharpe | Max DD | Trades | Win %")
    print("-" * 65)
    
    for name, result in results.items():
        print(f"{name:12} | {result['total_return']:6.1%} | {result['sharpe_ratio']:6.2f} | {result['max_drawdown']:6.1%} | {result['trade_count']:6} | {result['win_rate']:5.1%}")
    
    return results


if __name__ == "__main__":
    # Run main demo
    main_results = demo_week3_features()
    
    # Run comparison demo
    comparison_results = compare_strategies_demo()
    
    print("\n🎉 All demos completed successfully!")
    print("\nWeek 3 Features Implemented:")
    print("✅ Portfolio Accounting (NAV tracking, P&L, position management)")
    print("✅ Performance Metrics (12+ trading metrics)")
    print("✅ Report Generation (JSON, CSV, interactive charts)")
    print("✅ DuckDB Analytics Integration (foundation)")
    print("✅ Comprehensive Testing Suite")
    print("✅ Integration with Existing Week 1-2 Modules")