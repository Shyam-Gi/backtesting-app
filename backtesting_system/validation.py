"""
Bias Validation Module

Built-in checks for common backtesting pitfalls:
- Look-ahead bias detection
- Data quality validation
- Sanity checks vs. buy-and-hold baseline
- Realistic cost warnings
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import polars as pl

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check"""
    
    def __init__(self, name: str, passed: bool, message: str, severity: str = "warning"):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # "info", "warning", "error"
    
    def __str__(self):
        status = "✅" if self.passed else "❌"
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨"}.get(self.severity, "")
        return f"{severity_icon} {status} {self.name}: {self.message}"


class BiasValidator:
    """Validates backtest results for common biases and issues"""
    
    def __init__(self, warn_on_low_costs: bool = True):
        """
        Initialize validator
        
        Args:
            warn_on_low_costs: Whether to warn about unrealistic low costs
        """
        self.warn_on_low_costs = warn_on_low_costs
        self.results: List[ValidationResult] = []
    
    def validate_lookahead_bias(self, data: pl.DataFrame, signals: pl.DataFrame) -> ValidationResult:
        """
        Check for look-ahead bias in signals
        
        A signal should only use data available at that time.
        We check this by ensuring signals don't "predict" future price movements.
        
        Args:
            data: OHLCV data
            signals: Data with signal column
            
        Returns:
            ValidationResult
        """
        try:
            # Ensure both dataframes have timestamp
            if 'timestamp' not in data.columns or 'timestamp' not in signals.columns:
                return ValidationResult(
                    "Look-ahead Bias",
                    False,
                    "Missing timestamp column in data or signals",
                    "error"
                )
            
            # Join data and signals on timestamp
            merged = data.join(signals.select(['timestamp', 'signal']), on='timestamp', how='inner')
            
            if merged.height == 0:
                return ValidationResult(
                    "Look-ahead Bias",
                    False,
                    "No matching timestamps between data and signals",
                    "error"
                )
            
            # Check if signals lead to immediate price moves (potential look-ahead)
            # Calculate next day's return
            merged = merged.with_columns([
                pl.col('close').pct_change(1).alias('next_return')
            ])
            
            # Check correlation between signals and next returns
            # High positive correlation might indicate look-ahead
            signal_returns = merged.filter(pl.col('signal') != 0)
            
            if signal_returns.height < 10:
                return ValidationResult(
                    "Look-ahead Bias",
                    True,
                    "Insufficient signal data for validation",
                    "info"
                )
            
            # Calculate average next return after signals
            avg_return_after_buy = signal_returns.filter(pl.col('signal') == 1)['next_return'].mean()
            avg_return_after_sell = signal_returns.filter(pl.col('signal') == -1)['next_return'].mean()
            
            # Warning thresholds (these are heuristics)
            if avg_return_after_buy > 0.05:  # 5% average next-day return after buy signals
                return ValidationResult(
                    "Look-ahead Bias",
                    False,
                    f"Suspiciously high next-day return after buy signals: {avg_return_after_buy:.3f}",
                    "warning"
                )
            
            if avg_return_after_sell < -0.05:  # -5% average next-day return after sell signals
                return ValidationResult(
                    "Look-ahead Bias",
                    False,
                    f"Suspiciously low next-day return after sell signals: {avg_return_after_sell:.3f}",
                    "warning"
                )
            
            return ValidationResult(
                "Look-ahead Bias",
                True,
                "No obvious look-ahead bias detected",
                "info"
            )
            
        except Exception as e:
            return ValidationResult(
                "Look-ahead Bias",
                False,
                f"Validation failed: {str(e)}",
                "error"
            )
    
    def validate_data_quality(self, data: pl.DataFrame) -> ValidationResult:
        """
        Validate data quality
        
        Args:
            data: OHLCV data to validate
            
        Returns:
            ValidationResult
        """
        try:
            # Check required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = set(required_cols) - set(data.columns)
            if missing:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Missing columns: {missing}",
                    "error"
                )
            
            # Check for null values
            null_counts = {col: data[col].null_count() for col in required_cols}
            total_nulls = sum(null_counts.values())
            
            if total_nulls > 0:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Found {total_nulls} null values in data: {null_counts}",
                    "error"
                )
            
            # Check OHLC relationships
            invalid_high_low = data.filter(pl.col('high') < pl.col('low')).height
            if invalid_high_low > 0:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Found {invalid_high_low} rows where High < Low",
                    "error"
                )
            
            # Check for negative prices/volume
            negative_prices = data.filter(
                (pl.col('open') <= 0) | (pl.col('high') <= 0) | 
                (pl.col('low') <= 0) | (pl.col('close') <= 0)
            ).height
            
            if negative_prices > 0:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Found {negative_prices} rows with non-positive prices",
                    "error"
                )
            
            negative_volume = data.filter(pl.col('volume') < 0).height
            if negative_volume > 0:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Found {negative_volume} rows with negative volume",
                    "error"
                )
            
            # Check for duplicate timestamps
            duplicates = data.height - data.select('timestamp').n_unique()
            if duplicates > 0:
                return ValidationResult(
                    "Data Quality",
                    False,
                    f"Found {duplicates} duplicate timestamps",
                    "warning"
                )
            
            return ValidationResult(
                "Data Quality",
                True,
                f"Data quality looks good ({data.height} rows)",
                "info"
            )
            
        except Exception as e:
            return ValidationResult(
                "Data Quality",
                False,
                f"Validation failed: {str(e)}",
                "error"
            )
    
    def validate_realistic_costs(self, commission_bps: float, slippage_pct: float) -> ValidationResult:
        """
        Validate realistic trading costs
        
        Args:
            commission_bps: Commission in basis points
            slippage_pct: Slippage percentage
            
        Returns:
            ValidationResult
        """
        try:
            if not self.warn_on_low_costs:
                return ValidationResult(
                    "Realistic Costs",
                    True,
                    "Cost validation disabled",
                    "info"
                )
            
            warnings = []
            
            # Check commission (typical range: 1-10 bps for retail)
            if commission_bps < 1:
                warnings.append("Very low commission (typical: 1-10 bps)")
            elif commission_bps > 10:
                warnings.append("High commission (typical: 1-10 bps)")
            
            # Check slippage (typical range: 0.05% - 0.5% for liquid stocks)
            if slippage_pct < 0.0005:  # 0.05%
                warnings.append("Very low slippage (typical: 0.05% - 0.5%)")
            elif slippage_pct > 0.005:  # 0.5%
                warnings.append("High slippage (typical: 0.05% - 0.5%)")
            
            if warnings:
                return ValidationResult(
                    "Realistic Costs",
                    False,
                    "; ".join(warnings),
                    "warning"
                )
            
            return ValidationResult(
                "Realistic Costs",
                True,
                f"Costs look realistic (commission: {commission_bps} bps, slippage: {slippage_pct:.3%})",
                "info"
            )
            
        except Exception as e:
            return ValidationResult(
                "Realistic Costs",
                False,
                f"Validation failed: {str(e)}",
                "error"
            )
    
    def validate_vs_buy_and_hold(
        self, 
        strategy_returns: List[float], 
        buy_hold_returns: List[float]
    ) -> ValidationResult:
        """
        Validate strategy performance against buy-and-hold baseline
        
        Args:
            strategy_returns: Strategy daily returns
            buy_hold_returns: Buy-and-hold daily returns
            
        Returns:
            ValidationResult
        """
        try:
            if len(strategy_returns) != len(buy_hold_returns):
                return ValidationResult(
                    "Buy-and-Hold Comparison",
                    False,
                    "Strategy and buy-and-hold returns have different lengths",
                    "error"
                )
            
            if len(strategy_returns) < 30:  # Less than 1 month of data
                return ValidationResult(
                    "Buy-and-Hold Comparison",
                    True,
                    "Insufficient data for meaningful comparison",
                    "info"
                )
            
            # Calculate cumulative returns
            strategy_cum = 1.0
            buy_hold_cum = 1.0
            
            for s_r, bh_r in zip(strategy_returns, buy_hold_returns):
                strategy_cum *= (1 + s_r)
                buy_hold_cum *= (1 + bh_r)
            
            strategy_total_return = strategy_cum - 1
            buy_hold_total_return = buy_hold_cum - 1
            
            # Calculate outperformance
            outperformance = strategy_total_return - buy_hold_total_return
            
            # Generate message
            if outperformance > 0:
                message = f"Strategy outperformed buy-and-hold by {outperformance:.2%}"
            else:
                message = f"Strategy underperformed buy-and-hold by {-outperformance:.2%}"
            
            # Add context about total returns
            message += f" (Strategy: {strategy_total_return:.2%}, Buy&Hold: {buy_hold_total_return:.2%})"
            
            # Check for extreme underperformance
            if outperformance < -0.5:  # More than 50% underperformance
                return ValidationResult(
                    "Buy-and-Hold Comparison",
                    False,
                    f"{message} - Consider strategy effectiveness",
                    "warning"
                )
            
            return ValidationResult(
                "Buy-and-Hold Comparison",
                True,
                message,
                "info"
            )
            
        except Exception as e:
            return ValidationResult(
                "Buy-and-Hold Comparison",
                False,
                f"Validation failed: {str(e)}",
                "error"
            )
    
    def validate_trade_frequency(self, trades: List[Dict[str, Any]], data_points: int) -> ValidationResult:
        """
        Validate trade frequency
        
        Args:
            trades: List of trade dictionaries
            data_points: Number of data points (days) in backtest
            
        Returns:
            ValidationResult
        """
        try:
            if not trades:
                return ValidationResult(
                    "Trade Frequency",
                    False,
                    "No trades executed - strategy may be too conservative",
                    "warning"
                )
            
            trade_count = len(trades)
            trading_days = data_points
            trades_per_day = trade_count / trading_days
            trades_per_month = trades_per_day * 21  # ~21 trading days per month
            trades_per_year = trades_per_month * 12
            
            # Categorize trading frequency
            if trades_per_year < 2:
                freq_desc = "Very Low (< 2 trades/year)"
                severity = "warning"
            elif trades_per_year < 12:
                freq_desc = "Low (< 1 trade/month)"
                severity = "info"
            elif trades_per_year < 52:
                freq_desc = "Moderate (< 1 trade/week)"
                severity = "info"
            elif trades_per_year < 252:
                freq_desc = "High (< 1 trade/day)"
                severity = "info"
            else:
                freq_desc = "Very High (>= 1 trade/day)"
                severity = "warning"
            
            message = f"{trade_count} trades over {trading_days} days ({freq_desc})"
            
            return ValidationResult(
                "Trade Frequency",
                True,
                message,
                severity
            )
            
        except Exception as e:
            return ValidationResult(
                "Trade Frequency",
                False,
                f"Validation failed: {str(e)}",
                "error"
            )
    
    def validate_all(
        self,
        data: pl.DataFrame,
        signals: pl.DataFrame,
        trades: List[Dict[str, Any]],
        commission_bps: float,
        slippage_pct: float,
        strategy_returns: Optional[List[float]] = None,
        buy_hold_returns: Optional[List[float]] = None
    ) -> List[ValidationResult]:
        """
        Run all validations
        
        Args:
            data: OHLCV data
            signals: Data with signal column
            trades: List of trade dictionaries
            commission_bps: Commission in basis points
            slippage_pct: Slippage percentage
            strategy_returns: Optional strategy daily returns
            buy_hold_returns: Optional buy-and-hold daily returns
            
        Returns:
            List of ValidationResult objects
        """
        self.results = []
        
        # Run all validations
        self.results.append(self.validate_data_quality(data))
        self.results.append(self.validate_lookahead_bias(data, signals))
        self.results.append(self.validate_realistic_costs(commission_bps, slippage_pct))
        self.results.append(self.validate_trade_frequency(trades, len(data)))
        
        if strategy_returns is not None and buy_hold_returns is not None:
            self.results.append(self.validate_vs_buy_and_hold(strategy_returns, buy_hold_returns))
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.results:
            return {"total": 0, "passed": 0, "warnings": 0, "errors": 0, "infos": 0}
        
        passed = sum(1 for r in self.results if r.passed)
        warnings = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        infos = sum(1 for r in self.results if r.severity == "info")
        
        return {
            "total": len(self.results),
            "passed": passed,
            "warnings": warnings,
            "errors": errors,
            "infos": infos,
            "has_critical_issues": errors > 0
        }
    
    def print_results(self):
        """Print validation results in a formatted way"""
        if not self.results:
            print("🔍 No validation results to display")
            return
        
        summary = self.get_summary()
        
        print("\\n🔍 Bias Validation Results")
        print("=" * 50)
        
        for result in self.results:
            print(str(result))
        
        print("\\n📊 Summary")
        print("=" * 30)
        print(f"Total Checks: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"🚨 Errors: {summary['errors']}")
        print(f"ℹ️  Info: {summary['infos']}")
        
        if summary['has_critical_issues']:
            print("\\n🚨 CRITICAL ISSUES FOUND - Review backtest carefully!")
        elif summary['warnings'] > 0:
            print("\\n⚠️  WARNINGS - Consider reviewing these issues")
        else:
            print("\\n✅ Validation completed successfully!")