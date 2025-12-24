"""Backtesting system package."""

__version__ = "0.1.0"
__author__ = "Backtesting Team"

from backtesting_system.data_store import DataStore
from backtesting_system.data_loader import DataLoader
from backtesting_system.strategy import BaseStrategy
from backtesting_system.simulator import Simulator
from backtesting_system.accounting import Accounting
from backtesting_system.metrics import MetricsCalculator
from backtesting_system.reporting import BacktestReport
from backtesting_system.runner import BacktestConfig, run_backtest, run_backtest_with_report, run_multiple_backtests
from backtesting_system.validation import BiasValidator, ValidationResult

__all__ = [
    "DataStore", "DataLoader", "BaseStrategy", "Simulator", 
    "Accounting", "MetricsCalculator", "BacktestReport",
    "BacktestConfig", "run_backtest", "run_backtest_with_report", "run_multiple_backtests",
    "BiasValidator", "ValidationResult"
]
