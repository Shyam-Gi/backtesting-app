# Product Requirements Document: Stock Backtesting System (V1)

**Document Version:** 3.0  
**Created:** December 23, 2025  
**Status:** ✅ Approved for V1 Implementation  
**Approach:** Direct V1 build with scalable tech stack (Polars, DuckDB, joblib)

---

## 1. Problem Statement

**Current Pain Points:**
- New traders and investors cannot safely test trading strategies without risking real money
- Existing backtesting tools (Bloomberg, Refinitiv) are expensive and require licenses
- Free tools often lack transparency about assumptions and market realism
- Understanding which strategy works requires tedious manual calculations
- No easy way to compare multiple strategies side-by-side

**Opportunity:**
Build an open-source, educational backtesting system that allows users to:
- Test trading strategies on historical data
- Understand realistic execution (slippage, commissions, liquidity)
- Generate clear performance reports and metrics
- Learn why strategies succeed or fail
- Safely validate ideas before risking capital

---

## 2. Product Vision & Goals

**Vision:** A transparent, educational backtesting engine that helps traders validate strategies and understand market dynamics without financial risk.

**Primary Goals:**
1. Enable safe, repeatable testing of trading strategies on historical data
2. Provide realistic market simulation (commissions, slippage, position sizing)
3. Generate comprehensive performance metrics and visualizations
4. Support extensible strategy framework (custom strategies, plugin architecture)
5. Production-ready architecture (scalable, vectorized, stateless)

**Secondary Goals:**
- Build foundation for LLM-powered insights (explain strategy performance)
- Support live/paper trading connectors in future versions
- Enable strategy optimization and parameter grid search

---

## 3. Target Users & Use Cases

### User Persona (Single-User Focus)

**Primary User: Self-Directed Trader**
- Goal: Test trading strategies safely, compare strategies visually
- Technical skill: Intermediate Python, understands trading metrics
- Environment: Local machine, personal use only
- Workflow: Define strategy → Run backtest → View results in dashboard → Iterate

### Use Cases

| Use Case | Flow |
|----------|------|
| Test simple strategy | Define SMA crossover → Run backtest → View equity curve |
| Compare 2-3 strategies | Load multiple strategies → Run backtests → Compare metrics |
| Understand costs | Run strategy with/without slippage/commission → See impact |
| Iterate quickly | Tweak parameters → Re-run backtest → See results instantly |

---

## 4. Scope & Constraints

### In Scope (V1)

**Core Features:**
- Historical OHLCV data ingestion (stocks via yfinance)
- Strategy framework (entry/exit rules, position sizing)
- Execution simulator (slippage, commission)
- Portfolio accounting (NAV, positions, P&L)
- Performance metrics (12+ metrics: Sharpe, max drawdown, win rate, etc.)
- Trade logs and reporting
- Custom strategy plugin support (auto-discovery)

**Technology Stack (Scalable):**
- **Polars** (10x faster than Pandas; vectorized, columnar)
- **DuckDB** (100x faster analytics queries)
- **Joblib** (parallel backtests; 4x speedup on 4 cores)
- Streamlit dashboard (local deployment)
- Parquet data storage (compressed, columnar)

**User Interfaces:**
- CLI with batch mode support
- Streamlit web dashboard
- Jupyter notebook examples

**Quality Assurance:**
- 85%+ unit test coverage
- Integration tests (backend + UI with Playwright)
- E2E tests (parameter changes → different results)
- Bias validation (look-ahead, data quality, sanity checks)
- Type hints + mypy strict mode

### Out of Scope (V2+)

- Distributed compute (Dask/Ray) - in V3
- PostgreSQL/TimescaleDB migration - in V2
- Live trading connectors (Alpaca) - in V2
- Real-time market data - in V2
- Multi-user support, authentication, cloud deployment - in V3
- Cryptocurrency integration - future

### Constraints

| Constraint | Mitigation |
|-----------|-----------|
| Data quality | Use Yahoo Finance (free, reliable); validate & cache |
| API rate limits | Pre-download & cache OHLCV; batch requests |
| Realistic simulation | Model slippage, commission, position sizing |
| Look-ahead bias | Enforce causal signal checks; validation tests |
| Beginner usage | Clear docs, examples, safety warnings |

---

## 5. Feature Set

### V1 Core Features (8 Features)

#### 5.1 Data Loader
**Load historical OHLCV data with caching**

- Load daily OHLCV via yfinance for any US stock
- Any date range (default: 5 years)
- Cache in Parquet format (compressed, columnar)
- Timezone-aware timestamps
- Data validation (no NaN, monotonic, realistic ranges)
- < 500ms uncached, < 100ms cached load time

**Example:**
```python
loader = DataLoader()
df = loader.load(symbol="AAPL", start="2020-01-01", end="2024-12-31")
# Returns: DataFrame [timestamp, open, high, low, close, volume]
```

---

#### 5.2 Strategy API (Pluggable)
**Simple, extensible strategy framework**

- Inherit from `BaseStrategy` base class
- Implement `generate_signals()` method
- Fixed position sizing support
- 3+ built-in example strategies (SMA, Momentum, BuyHold)
- Custom strategy auto-discovery (no core code changes)
- Full access to indicators and calculations

**Example Built-in Strategy:**
```python
from backtesting_system.strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df):
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # BUY
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # SELL
        return df
```

**Example Custom Strategy:**
```python
# File: my_strategies.py
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, ma_period=20, threshold=0.05):
        self.ma_period = ma_period
        self.threshold = threshold
    
    def generate_signals(self, df):
        df['ma'] = df['close'].rolling(self.ma_period).mean()
        df['distance'] = (df['close'] - df['ma']) / df['ma']
        df['signal'] = 0
        df.loc[df['distance'] < -self.threshold, 'signal'] = 1  # BUY
        df.loc[df['distance'] >= 0, 'signal'] = -1  # SELL
        return df
```

---

#### 5.3 Execution Simulator
**Simulate realistic order execution**

- Execute trades at signal prices
- Configurable commission (bps or fixed)
- Configurable slippage (percentage or fixed)
- No shorting (cash-limited buys only)
- Trade execution log
- Unit-tested against manual calculations

**Example:**
```python
simulator = Simulator(
    initial_cash=100000,
    commission_bps=5,      # 5 basis points
    slippage_pct=0.001,    # 0.1% slippage
    position_size_pct=1.0  # 100% of cash
)
```

---

#### 5.4 Portfolio Accounting
**Track positions, cash, and P&L**

- Daily NAV (net asset value) calculation
- Open positions tracking (entry price, entry date, current value)
- Realized and unrealized P&L
- Transaction log (timestamp, side, price, qty, fees, slippage)
- NAV validation: Cash + Sum(Position values)

---

#### 5.5 Performance Metrics & Reporting
**Calculate standard trading metrics**

**Metrics (12+):**
- Returns: Total return (%), CAGR, daily returns
- Risk: Volatility, max drawdown, drawdown duration
- Risk-Adjusted: Sharpe, Sortino, Calmar ratios
- Trade Stats: Win rate (%), avg win/loss, profit factor, expectancy
- Efficiency: Trade count, turnover (%), avg holding period
- Benchmark: vs. buy-and-hold

**Outputs:**
- JSON summary report
- CSV trade log
- Equity curve + drawdown plots (Plotly)
- Metrics < 100ms (DuckDB analytics)

---

#### 5.6 Backtesting Runner
**Orchestrate end-to-end pipeline**

- Accept: strategy, symbol, date range, initial capital
- Execute: load → signal → simulate → account → metrics
- Stateless pure function (reproducible, parallelizable)
- < 1 second for full 5-year daily backtest
- Clear error messages

**Example CLI:**
```bash
python run_backtest.py \
  --strategy SMAStrategy \
  --symbol AAPL \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --initial_cash 100000 \
  --output results/backtest.json
```

---

#### 5.7 Web UI Dashboard (Streamlit)
**Interactive local dashboard**

**Pages:**
1. **Strategy Runner** - Select strategy, symbol, dates, parameters → Run
2. **Results Dashboard** - Equity curve, drawdown, trade markers, metrics
3. **Strategy Comparison** - Side-by-side comparison of 2-3 backtests
4. **Trade Log** - Detailed trade table with entry/exit/P&L

**Features:**
- Runs on localhost:8501
- Parameter input fields
- Interactive charts (Plotly)
- Export JSON/CSV
- < 1 second load time (Redis-cached results optional)

---

#### 5.8 Bias Validation & Testing
**Built-in checks for common pitfalls**

**Validation:**
- Look-ahead bias (signals only use past data)
- Data quality (missing data, duplicates, realistic ranges)
- Sanity checks (compare vs. buy-and-hold baseline)
- Reproducibility (log data version, strategy code, seed)
- Realistic costs (warn if slippage/commission too low)

---

### V2+ Features (Future Phases)

**V2 (3 months):**
- Parameter grid search & walk-forward optimization
- LLM integration: explain strategy performance
- PostgreSQL backend option
- Advanced metrics (factor attribution)

**V3 (6 months):**
- Distributed compute (Dask/Ray)
- Paper trading connector (Alpaca)
- Multi-asset portfolio backtesting
- Cloud deployment

---

## 6. UI & UX Design

### Streamlit Dashboard Mockups

#### Page 1: Strategy Runner
```
BACKTEST SYSTEM - Strategy Runner
═════════════════════════════════════════

Strategy Configuration
┌─────────────────────────────────────┐
│ Strategy:      [SMA Crossover ▼]    │
│ Symbol:        [AAPL         ▼]    │
│ Start:         [2020-01-01   ]     │
│ End:           [2024-12-31   ]     │
│ Initial Capital: [$100,000]        │
│                                     │
│ Parameters:                         │
│  • Fast MA Period:  [10]           │
│  • Slow MA Period:  [50]           │
│  • Commission (bps): [5]           │
│  • Slippage (%):     [0.1]         │
│                                     │
│        [RUN BACKTEST]  [RESET]    │
└─────────────────────────────────────┘

Status: Ready
```

#### Page 2: Results Dashboard
```
RESULTS: SMA Crossover on AAPL (2020-2024)
═════════════════════════════════════════════

KEY METRICS          │  EQUITY CURVE
──────────────────   │  ──────────────
Total Return: +45.2% │  $150K
CAGR:          9.8%  │     ╱╲  ╱╲
Sharpe:        1.23  │   ╱  ╲╱  ╲╱
Max Drawdown: -12.5% │ ╱              ╲
Win Rate:      55%   │               ╲
Profit Factor:  1.8  │ $100K          ╲
Trades:         42   │                  ╲
                     │ 2020  2022  2024

[Export JSON] [Export CSV] [Compare] [New Test]
```

#### Page 3 & 4: Trade Log & Comparison (similarly structured)

---

## 7. Data Architecture

### Data Sources

| Source | Type | Cost | Coverage |
|--------|------|------|----------|
| Yahoo Finance | Free | $0 | US stocks, 50+ years |
| Polygon.io | Paid | $99–$1999/mo | Tick-level, advanced |
| IEX Cloud | Paid | $99–$5000/mo | Real-time, detailed |

**V1 Plan:** Use yfinance for stocks; cache locally in Parquet.

### Data Storage Structure

```
llm-learnings/
├── data/
│   ├── raw/                      # Parquet OHLCV files
│   │   ├── AAPL_daily.parquet
│   │   ├── MSFT_daily.parquet
│   │   └── metadata.json         # Cache manifest
│   └── .gitignore
├── results/                      # Backtest outputs
│   ├── backtest_*.json
│   ├── trades_*.csv
│   └── equity_curve_*.png
└── backtesting_system/           # Core package
```

---

## 8. Technical Architecture

### Module Breakdown

```
backtesting_system/
├── data_loader.py           # Load/cache OHLCV (Parquet)
├── strategy.py              # BaseStrategy + examples
├── simulator.py             # Order execution
├── accounting.py            # NAV, positions, P&L
├── metrics.py               # Performance calculations
├── runner.py                # Pipeline orchestrator
├── validation.py            # Bias checks
├── cli.py                   # Command-line interface
├── app.py                   # Streamlit dashboard
├── utils.py                 # Helpers
├── strategies/              # Example strategies
│   ├── sma_crossover.py
│   ├── momentum.py
│   ├── buy_hold.py
│   └── README.md
├── tests/
│   ├── test_data_loader.py
│   ├── test_simulator.py
│   ├── test_metrics.py
│   └── ...
└── __init__.py
```

### V1 Architecture Principles

#### 1. Vectorization-First
- **Rule:** No `for` loops in backtest logic
- **Use:** Polars expressions, NumPy/DuckDB operations
- **Benefit:** 10x performance; easy migration to Dask/Ray later

#### 2. Stateless Functions
- **Rule:** Each backtest = pure function (same input → same output)
- **Benefit:** Thread-safe, cacheable, parallelizable
- **Example:** `backtest(strategy, symbol, dates) → results`

#### 3. Config-Driven
- **Rule:** All parameters in YAML/JSON (no hardcoding)
- **Benefit:** Reproducible, batch-friendly, Airflow-ready

#### 4. Data Abstraction
- **Rule:** `DataStore` interface supports Parquet ↔ DuckDB ↔ PostgreSQL
- **Benefit:** Swap backends without changing logic

### Key Dependencies

```ini
# Core Computation (Scalable)
polars>=0.19.0             # 10x faster than Pandas
numpy>=1.24.0              # Numerical operations
pyarrow>=13.0.0            # Parquet (Polars native)
duckdb>=0.9.0              # Analytics DB; < 50ms queries
joblib>=1.3.0              # Parallel backtests

# Data & Visualization
yfinance>=0.2.28           # Free historical data
streamlit>=1.28.0          # Web dashboard
plotly>=5.13.0             # Interactive charts

# Quality & Testing
pytest>=7.4.0              # Unit testing
playwright>=1.40.0          # E2E UI testing
pytest-playwright>=0.4.0     # Playwright integration
mypy>=1.6.0                # Type checking
black>=23.10.0             # Formatting
flake8>=6.1.0              # Linting

# Config & Environment
pydantic>=2.0.0            # Data validation
python-dotenv>=1.0.0       # .env loading

# Optional (V2+)
redis>=5.0.0               # Result caching
psycopg2                   # PostgreSQL (V2)
```

### Data Flow

```
User Input (strategy, symbol, dates)
    ↓
DataLoader.load() → Polars DataFrame
    ↓
Strategy.generate_signals() → Vectorized signals
    ↓
Simulator.execute_trades() → Trade list
    ↓
Accounting.process_trades() → NAV, P&L
    ↓
Metrics.calculate() → DuckDB analytics
    ↓
Report + Dashboard
```

---

## 9. Scalability Considerations

### V1 → V2 → V3 Evolution

| Bottleneck | V1 | V2 | V3 |
|-----------|----|----|-----|
| Data Loading | Parquet (local) | DuckDB | PostgreSQL |
| Backtest Speed | Polars vectorized | + Numba JIT | Distributed |
| Parallelization | joblib (4 cores) | joblib (16 cores) | Dask/Ray (100s) |
| Result Storage | JSON/CSV | SQLite metadata | PostgreSQL |
| UI Performance | Streamlit | + Redis cache | Dash/FastAPI |

### Scalable Design Patterns

1. **Vectorization-First** - No loops; use expressions
2. **Cacheable Pipeline** - Idempotent transforms; cache results
3. **Configuration Over Code** - YAML for all parameters
4. **Database-Agnostic** - DataStore interface
5. **Parallel-Ready** - Stateless functions

---

## 10. Success Criteria

### Functional (V1)

- [x] Data loading: < 500ms uncached, < 100ms cached
- [x] 3+ example strategies + custom plugin support
- [x] Execution simulator with realistic costs
- [x] 12+ performance metrics calculated
- [x] Backtest < 1 second (5-year daily)
- [x] 4 parallel backtests < 2 seconds (joblib)
- [x] Streamlit dashboard + CLI interface
- [x] 85%+ unit test coverage

### Scalability (Built-in V1)

- [x] Vectorized computation (no for-loops)
- [x] Stateless backtests (pure functions)
- [x] Data abstraction layer (swappable backends)
- [x] Batch mode CLI (`--batch backtests.yaml`)
- [x] Config-driven (YAML reproducibility)

### Quality

- [x] Type hints + mypy strict
- [x] Comprehensive tests
- [x] Clear error messages
- [x] Performance benchmarks
- [x] Bias validation built-in

---

## 11. Implementation Roadmap

### Phase 1: V1 (4 weeks)

**Week 1: Foundation & Data (Polars + Parquet)**
- Project setup, git, requirements.txt
- DataStore interface
- DataLoader with Polars (lazy evaluation)
- Parquet caching
- Unit tests
- **Deliverable:** Load 5-year AAPL in < 500ms

**Week 2: Strategy & Execution (Vectorized)**
- BaseStrategy class
- SMA & Momentum examples (Polars expressions)
- Simulator (vectorized batch execution)
- Integration tests
- **Deliverable:** Full backtest < 1 second

**Week 3: Accounting & Metrics (DuckDB)**
- Accounting (Polars position tracking)
- Metrics (DuckDB analytics < 50ms)
- Report generation (JSON, CSV)
- Plotting (Plotly equity curve, drawdown)
- **Deliverable:** Full report + metrics < 500ms

**Week 4: Runner, CLI, Dashboard (Joblib)**
- Runner orchestrator (stateless, parallelizable)
- CLI with batch mode (`--batch backtests.yaml`)
- Streamlit dashboard (4 pages)
- **Integration testing with Playwright**
  - UI parameter changes → different backtest results
  - Multi-page navigation testing
  - Data flow validation (input → output)
  - Error handling verification
- Bias validation tests
- Documentation (README, examples)
- **Deliverable:** 4 backtests in 2 seconds (joblib 4 cores) + 95% test coverage

---

## 12. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Incorrect P&L | Unit tests vs. manual calcs; validation |
| Look-ahead bias | Causal checks; test detection |
| Data quality | Validate OHLCV; version tracking |
| Slow backtest | Vectorized from start; profile early |
| Scalability bottleneck | Design for scale now (stateless, abstracted) |
| User treats as financial advice | Clear disclaimers; educational framing |

---

## 13. Open Questions

1. **Support shorting?** (MVP: no; V2: yes)
2. **Multiple assets?** (V1: single; V2: portfolio)
3. **Intraday trading?** (V1: daily only; V2: hourly/minute)
4. **Corporate actions?** (V1: adjusted close; V2: explicit)
5. **Distributed framework?** (Dask vs. Ray for V3)
6. **Database at scale?** (PostgreSQL + TimescaleDB vs. ClickHouse)

---

## 14. Glossary & References

**Key Terms:**
- **OHLCV:** Open, High, Low, Close, Volume
- **Backtest:** Historical strategy simulation
- **Sharpe Ratio:** Return / volatility (risk-adjusted)
- **Max Drawdown:** Peak-to-trough loss
- **Slippage:** Expected vs. actual fill price
- **Commission:** Fee per trade

**Resources:**
- [Polars Docs](https://www.pola-rs.org/)
- [DuckDB Docs](https://duckdb.org/)
- [Backtrader](https://www.backtrader.com/)
- [Investopedia: Backtesting](https://www.investopedia.com/terms/b/backtesting.asp)

---

## 15. Sign-Off

**Prepared by:** AI Assistant  
**Date:** December 23, 2025  
**Status:** ✅ Approved for V1 Implementation  
**Approach:** Direct V1 build (Polars + DuckDB + joblib)

### Why V1-Direct (No MVP Refactor Cycle)

- ✅ **4 weeks** (not 8 weeks with MVP refactor)
- ✅ **Polars from day 1** (no Pandas → Polars migration)
- ✅ **Zero tech debt** (production-ready from start)
- ✅ **10x faster** backtest execution
- ✅ **Scalable architecture** (vectorized, stateless, abstracted)

### Next Steps

1. ✅ Review & approve PRD
2. ✅ Confirm V1 architecture principles
3. **Begin Week 1 Implementation:**
   - DataStore interface + DataLoader (Polars)
   - Unit tests with performance assertions
   - Ready to code

---

**Ready to build V1. No more planning. Let's execute.** 🚀
