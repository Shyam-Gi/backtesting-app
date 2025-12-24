"""
Backtest Runner - orchestrates end-to-end pipeline

Stateless pure function (reproducible, parallelizable)
< 1 second for full 5-year daily backtest
"""

from datetime import datetime
from typing import Dict, Any, Type, Optional
import logging

from .data_loader import DataLoader
from .accounting import Accounting
from .metrics import MetricsCalculator
from .reporting import BacktestReport
from .simulator import Simulator
from .strategy import BaseStrategy


logger = logging.getLogger(__name__)


class BacktestConfig:
    """Configuration for a single backtest"""
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_params: Dict[str, Any],
        symbol: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 100000,
        commission_bps: float = 5.0,
        slippage_pct: float = 0.001,
        position_size_pct: float = 1.0,
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'strategy_name': self.strategy_class.__name__,
            'strategy_params': self.strategy_params,
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_cash': self.initial_cash,
            'commission_bps': self.commission_bps,
            'slippage_pct': self.slippage_pct,
            'position_size_pct': self.position_size_pct,
        }


def run_backtest(config: BacktestConfig) -> Dict[str, Any]:
    """
    Pure function to execute a complete backtest
    
    Args:
        config: BacktestConfig containing all parameters
        
    Returns:
        Dictionary with complete results including trades, metrics, and metadata
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Load data
        logger.info(f"Loading data for {config.symbol} ({config.start_date} to {config.end_date})")
        loader = DataLoader()
        data = loader.load(config.symbol, config.start_date, config.end_date)
        
        # Step 2: Generate signals
        logger.info(f"Generating signals with {config.strategy_class.__name__}")
        strategy = config.strategy_class(**config.strategy_params)
        signals = strategy.generate_signals(data)
        
        # Step 3: Execute trades
        logger.info("Executing trades")
        simulator = Simulator(
            initial_cash=config.initial_cash,
            commission_bps=config.commission_bps,
            slippage_pct=config.slippage_pct,
            position_size_pct=config.position_size_pct,
        )
        sim_result = simulator.execute(signals)
        
        # Step 4: Portfolio accounting
        logger.info("Processing portfolio accounting")
        accounting = Accounting(initial_cash=config.initial_cash)
        accounting.process_trades(sim_result['trades'], signals)
        
        # Step 5: Calculate metrics
        logger.info("Calculating performance metrics")
        calculator = MetricsCalculator(risk_free_rate=0.02)
        nav_history = accounting.get_nav_history()
        metrics = calculator.calculate(nav_history, sim_result['trades'])
        
        # Step 6: Build result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            # Configuration
            'config': config.to_dict(),
            
            # Basic results
            'trades': sim_result['trades'],
            'final_portfolio_value': sim_result['final_portfolio_value'],
            'total_return': sim_result['total_return'],
            'num_trades': sim_result['num_trades'],
            
            # Accounting data
            'nav_history': nav_history,
            'positions': accounting.get_positions(),
            'cash_history': accounting.get_cash_history(),
            
            # Performance metrics
            'metrics': metrics,
            
            # Metadata
            'execution_time_seconds': execution_time,
            'data_points': len(signals),
            'success': True,
            'error': None,
        }
        
        logger.info(f"Backtest completed in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return {
            'config': config.to_dict(),
            'trades': [],
            'final_portfolio_value': config.initial_cash,
            'total_return': 0.0,
            'num_trades': 0,
            'nav_history': [],
            'positions': {},
            'cash_history': [],
            'metrics': {},
            'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
            'data_points': 0,
            'success': False,
            'error': str(e),
        }


def run_backtest_with_report(
    config: BacktestConfig,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run backtest and generate full report
    
    Args:
        config: BacktestConfig containing all parameters
        output_dir: Directory to save reports
        
    Returns:
        Dictionary with results and file paths
    """
    # Run backtest
    result = run_backtest(config)
    
    if not result['success']:
        return {**result, 'report_files': {}}
    
    # Generate report
    accounting = Accounting(initial_cash=config.initial_cash)
    if result['trades']:
        # Re-create accounting state for report generation
        loader = DataLoader()
        data = loader.load(config.symbol, config.start_date, config.end_date)
        strategy = config.strategy_class(**config.strategy_params)
        signals = strategy.generate_signals(data)
        accounting.process_trades(result['trades'], signals)
    
    report = BacktestReport(output_dir=output_dir)
    report_files = report.generate_full_report(
        accounting=accounting,
        metrics=result['metrics'],
        trades=result['trades'],
        strategy_name=config.strategy_class.__name__,
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        initial_cash=config.initial_cash
    )
    
    result['report_files'] = report_files
    return result


def run_multiple_backtests(configs: list[BacktestConfig]) -> list[Dict[str, Any]]:
    """
    Run multiple backtests in parallel
    
    Args:
        configs: List of BacktestConfig objects
        
    Returns:
        List of result dictionaries
    """
    from joblib import Parallel, delayed
    
    logger.info(f"Running {len(configs)} backtests in parallel")
    
    # Run in parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_backtest)(config) for config in configs
    )
    
    successful = sum(1 for r in results if r['success'])
    logger.info(f"Completed {successful}/{len(configs)} backtests successfully")
    
    return results