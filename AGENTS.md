# AGENTS.md - Stock Backtesting System

## Build/Test Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=backtesting_system

# Code quality (run in order)
black backtesting_system/ tests/
flake8 backtesting_system/ tests/
mypy backtesting_system/
```

## Code Style Guidelines
- **Type hints**: Full type annotations required for all functions
- **Docstrings**: Args, Returns, Raises documented for public methods
- **Imports**: Group stdlib, third-party, local imports with blank lines
- **Formatting**: Black (line length 88), no trailing whitespace
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **DataFrames**: Polars only (no Pandas in production code)
- **Error handling**: ValueError with descriptive messages for validation
- **Performance**: <100ms cached loads, <500ms uncached downloads
- **Testing**: 100% coverage for core modules, use fixtures for setup

## Architecture Principles
- Vectorized operations only (no for-loops in backtest logic)
- Stateless functions for thread safety
- Abstract interfaces for swappable backends (DataStore pattern)
- Config-driven with Pydantic validation