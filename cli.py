"""
CLI Interface for Backtesting System

Supports both single backtest and batch processing mode
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

from backtesting_system import (
    BacktestConfig, run_backtest, run_backtest_with_report, run_multiple_backtests,
    DataLoader
)
from strategies.sma_strategy import SMAStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy


# Strategy registry
STRATEGIES = {
    'SMA': SMAStrategy,
    'Momentum': MomentumStrategy,
    'MeanReversion': MeanReversionStrategy,
}


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_strategy_params(strategy_name: str, params: List[str]) -> Dict[str, Any]:
    """
    Parse strategy parameters from command line arguments
    
    Example: --param fast_period=10 --param slow_period=50
    """
    if not params:
        return {}
    
    parsed = {}
    for param in params:
        if '=' not in param:
            raise ValueError(f"Invalid parameter format: {param}. Use key=value")
        
        key, value = param.split('=', 1)
        
        # Try to convert to appropriate type
        if value.lower() in ('true', 'false'):
            parsed[key] = value.lower() == 'true'
        else:
            try:
                # Try integer first, then float
                if '.' not in value:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
    
    return parsed


def run_single_backtest(args):
    """Run a single backtest from command line arguments"""
    logger = logging.getLogger(__name__)
    
    # Validate strategy
    if args.strategy not in STRATEGIES:
        print(f"Error: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {list(STRATEGIES.keys())}")
        sys.exit(1)
    
    # Parse strategy parameters
    strategy_params = parse_strategy_params(args.strategy, args.param)
    
    # Create configuration
    config = BacktestConfig(
        strategy_class=STRATEGIES[args.strategy],
        strategy_params=strategy_params,
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
        commission_bps=args.commission_bps,
        slippage_pct=args.slippage_pct,
        position_size_pct=args.position_size_pct,
    )
    
    logger.info(f"Running single backtest for {args.strategy} on {args.symbol}")
    
    # Run backtest
    if args.output:
        result = run_backtest_with_report(config, args.output)
        
        if result['success']:
            print(f"\\n✅ Backtest completed successfully!")
            print(f"📊 Results saved to: {args.output}")
            for file_type, file_path in result['report_files'].items():
                print(f"   {file_type}: {file_path}")
        else:
            print(f"❌ Backtest failed: {result['error']}")
            sys.exit(1)
    else:
        result = run_backtest(config)
        
        if result['success']:
            print(f"\\n✅ Backtest completed successfully!")
            print(f"📈 Final Portfolio: ${result['final_portfolio_value']:,.2f}")
            print(f"📊 Total Return: {result['total_return']:.2%}")
            print(f"🔄 Number of Trades: {result['num_trades']}")
            print(f"⏱️  Execution Time: {result['execution_time_seconds']:.2f}s")
            
            # Print key metrics
            if result['metrics']:
                metrics = result['metrics']
                print(f"\\n📊 Performance Metrics:")
                if 'sharpe_ratio' in metrics:
                    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                if 'max_drawdown' in metrics:
                    print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
                if 'win_rate' in metrics:
                    print(f"   Win Rate: {metrics['win_rate']:.2%}")
        else:
            print(f"❌ Backtest failed: {result['error']}")
            sys.exit(1)


def run_batch_backtest(args):
    """Run batch backtests from configuration file"""
    logger = logging.getLogger(__name__)
    
    # Load configuration file
    try:
        with open(args.batch, 'r') as f:
            if args.batch.endswith('.yaml') or args.batch.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load configuration file: {e}")
        sys.exit(1)
    
    # Validate configuration structure
    if 'backtests' not in config_data:
        print("Error: Configuration file must contain 'backtests' array")
        sys.exit(1)
    
    # Create backtest configurations
    configs = []
    for i, backtest_config in enumerate(config_data['backtests']):
        try:
            strategy_name = backtest_config['strategy']
            if strategy_name not in STRATEGIES:
                print(f"Error: Unknown strategy '{strategy_name}' in backtest #{i+1}")
                sys.exit(1)
            
            config = BacktestConfig(
                strategy_class=STRATEGIES[strategy_name],
                strategy_params=backtest_config.get('params', {}),
                symbol=backtest_config['symbol'],
                start_date=backtest_config['start_date'],
                end_date=backtest_config['end_date'],
                initial_cash=backtest_config.get('initial_cash', 100000),
                commission_bps=backtest_config.get('commission_bps', 5.0),
                slippage_pct=backtest_config.get('slippage_pct', 0.001),
                position_size_pct=backtest_config.get('position_size_pct', 1.0),
            )
            configs.append(config)
            
        except KeyError as e:
            print(f"Error: Missing required key {e} in backtest #{i+1}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Invalid configuration in backtest #{i+1}: {e}")
            sys.exit(1)
    
    logger.info(f"Running {len(configs)} backtests in batch mode")
    
    # Run all backtests
    results = run_multiple_backtests(configs)
    
    # Output results
    successful = sum(1 for r in results if r['success'])
    print(f"\\n📊 Batch Results: {successful}/{len(results)} backtests completed successfully")
    
    # Print summary
    print("\\n📋 Summary:")
    for i, (config, result) in enumerate(zip(configs, results)):
        status = "✅" if result['success'] else "❌"
        strategy_name = config.strategy_class.__name__
        symbol = config.symbol
        
        if result['success']:
            return_pct = result['total_return'] * 100
            trades = result['num_trades']
            print(f"   {status} {strategy_name} on {symbol}: {return_pct:+.1f}% ({trades} trades)")
        else:
            print(f"   {status} {strategy_name} on {symbol}: {result['error']}")
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        results_data = {
            'timestamp': str(args.start),  # Using start as placeholder for timestamp
            'total_backtests': len(results),
            'successful_backtests': successful,
            'results': [
                {
                    'config': r['config'],
                    'success': r['success'],
                    'final_portfolio_value': r['final_portfolio_value'],
                    'total_return': r['total_return'],
                    'num_trades': r['num_trades'],
                    'execution_time_seconds': r['execution_time_seconds'],
                    'metrics': r.get('metrics', {}),
                    'error': r.get('error'),
                }
                for r in results
            ]
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\\n💾 Results saved to: {output_path}")
        except Exception as e:
            print(f"Error: Failed to save results: {e}")
    
    # Exit with error code if any backtests failed
    if successful < len(results):
        sys.exit(1)


def list_symbols():
    """List available symbols from cached data"""
    loader = DataLoader()
    
    # Check data directory
    data_dir = Path(loader.cache_dir)
    if not data_dir.exists():
        print("No cached data found. Run a backtest first to download data.")
        return
    
    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No cached data found.")
        return
    
    symbols = [f.stem for f in parquet_files]
    symbols.sort()
    
    print(f"📊 Found {len(symbols)} cached symbols:")
    for symbol in symbols:
        print(f"   {symbol}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Stock Backtesting System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single backtest
  python cli.py --strategy SMA --symbol AAPL --start 2020-01-01 --end 2024-12-31

  # With strategy parameters
  python cli.py --strategy SMA --symbol AAPL --start 2020-01-01 --end 2024-12-31 \\
                --param fast_period=10 --param slow_period=50

  # Batch backtest
  python cli.py --batch config/backtests.yaml --output results/batch_results.json

  # List available symbols
  python cli.py --list-symbols
        """
    )
    
    # Main arguments
    parser.add_argument('--strategy', type=str, help='Strategy name (SMA, Momentum, MeanReversion)')
    parser.add_argument('--symbol', type=str, help='Stock symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    
    # Strategy parameters
    parser.add_argument('--param', type=str, action='append', default=[], 
                       help='Strategy parameter in key=value format (can be used multiple times)')
    
    # Simulation parameters
    parser.add_argument('--initial-cash', type=float, default=100000, 
                       help='Initial cash amount (default: 100000)')
    parser.add_argument('--commission-bps', type=float, default=5.0, 
                       help='Commission in basis points (default: 5.0)')
    parser.add_argument('--slippage-pct', type=float, default=0.001, 
                       help='Slippage percentage (default: 0.001)')
    parser.add_argument('--position-size-pct', type=float, default=1.0, 
                       help='Position size as percentage of cash (default: 1.0)')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output directory for reports')
    parser.add_argument('--batch', type=str, help='Batch configuration file (JSON or YAML)')
    
    # Utility options
    parser.add_argument('--list-symbols', action='store_true', help='List available cached symbols')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle utility commands
    if args.list_symbols:
        list_symbols()
        return
    
    # Validate arguments
    if not args.batch and not (args.strategy and args.symbol and args.start and args.end):
        print("Error: Must specify either --batch or (--strategy --symbol --start --end)")
        parser.print_help()
        sys.exit(1)
    
    # Run appropriate mode
    if args.batch:
        run_batch_backtest(args)
    else:
        run_single_backtest(args)


if __name__ == "__main__":
    main()