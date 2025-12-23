"""Unit tests for backtesting simulator."""

import pytest
import polars as pl
import pandas as pd
from backtesting_system.simulator import Simulator, Trade


class TestSimulator:
    """Test Simulator class."""

    @pytest.fixture
    def sample_data_with_signals(self):
        """Create sample OHLCV data with trading signals."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-20", freq="D")
        close_prices = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99,
                      98, 99, 100, 101, 102, 103, 104, 105, 106, 105]
        
        # Create signals: BUY at start, SELL at peak, BUY again at dip
        signals = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 
                  1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
        
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [p - 0.2 for p in close_prices],
            'high': [p + 0.5 for p in close_prices],
            'low': [p - 0.5 for p in close_prices],
            'close': close_prices,
            'volume': [1000000] * len(close_prices),
            'signal': signals
        })
        return df

    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return Simulator(
            initial_cash=10000,
            commission_bps=5.0,  # 5 bps = 0.05%
            slippage_pct=0.001,  # 0.1% slippage
            position_size_pct=1.0
        )

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        sim = Simulator(
            initial_cash=50000,
            commission_bps=10.0,
            slippage_pct=0.002,
            position_size_pct=0.8
        )
        
        assert sim.initial_cash == 50000
        assert sim.cash == 50000
        assert sim.position == 0
        assert sim.commission_bps == 10.0
        assert sim.slippage_pct == 0.002
        assert sim.position_size_pct == 0.8
        assert len(sim.trades) == 0

    def test_simulator_rates(self, simulator):
        """Test commission and slippage rate calculations."""
        assert simulator.commission_rate == 0.0005  # 5 bps
        assert simulator.slippage_rate == 0.001

    def test_execute_no_signals(self, simulator):
        """Test execution with no trading signals."""
        # Data with only HOLD signals
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5,
            'signal': [0, 0, 0, 0, 0]
        })
        
        result = simulator.execute(df)
        
        # No trades should be executed
        assert result['num_trades'] == 0
        assert result['final_cash'] == simulator.initial_cash
        assert result['final_position'] == 0
        assert result['total_return'] == 0.0

    def test_execute_buy_signals(self, simulator):
        """Test execution of BUY signals."""
        # Data with initial BUY signal
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000000] * 10,
            'signal': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]  # BUY then SELL
        })
        
        result = simulator.execute(df)
        
        # Should have executed trades
        assert result['num_trades'] >= 2  # At least one BUY and one SELL
        assert len(simulator.trades) >= 2
        
        # Check first trade is BUY
        first_trade = simulator.trades[0]
        assert first_trade.side == 'BUY'
        assert first_trade.commission > 0
        assert first_trade.slippage > 0  # Buy price higher than market

    def test_commission_calculation(self, simulator):
        """Test commission calculation."""
        # Create simple BUY trade
        dates = pd.date_range(start="2024-01-01", end="2024-01-03", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000000] * 3,
            'signal': [1, -1, -1]  # BUY then SELL
        })
        
        result = simulator.execute(df)
        
        # Check commission is calculated correctly
        buy_trade = None
        sell_trade = None
        
        for trade in simulator.trades:
            if trade.side == 'BUY':
                buy_trade = trade
            elif trade.side == 'SELL':
                sell_trade = trade
        
        assert buy_trade is not None
        assert sell_trade is not None
        
        # Commission = value * rate
        expected_buy_commission = buy_trade.quantity * buy_trade.price * 0.0005
        assert abs(buy_trade.commission - expected_buy_commission) < 0.01

    def test_slippage_calculation(self, simulator):
        """Test slippage calculation."""
        # Create simple trades
        dates = pd.date_range(start="2024-01-01", end="2024-01-03", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000000] * 3,
            'signal': [1, -1, -1]  # BUY then SELL
        })
        
        simulator.execute(df)
        
        buy_trade = next((t for t in simulator.trades if t.side == 'BUY'), None)
        sell_trade = next((t for t in simulator.trades if t.side == 'SELL'), None)
        
        # BUY: should pay higher price (positive slippage)
        assert buy_trade.slippage > 0
        assert buy_trade.price > 100  # Higher than close price
        
        # SELL: should receive lower price (negative slippage)
        # Note: slippage is calculated as (price - trade_price) * shares
        # Since trade_price < price, the result is positive per share, but negative total
        expected_slippage = (101 - sell_trade.price) * sell_trade.quantity
        assert abs(sell_trade.slippage - expected_slippage) < 0.01
        assert sell_trade.price < 101  # Lower than close price

    def test_portfolio_value_calculation(self, simulator):
        """Test final portfolio value calculation."""
        # Create trade that ends with position
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5,
            'signal': [1, 1, 1, 1, -1]  # BUY until last day
        })
        
        result = simulator.execute(df)
        
        # Portfolio value = cash + position_value
        final_price = 104
        expected_value = result['final_cash'] + (result['final_position'] * final_price)
        assert abs(result['final_portfolio_value'] - expected_value) < 0.01

    def test_reset_functionality(self, simulator):
        """Test simulator reset functionality."""
        # Execute some trades first
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5,
            'signal': [1, -1, 1, -1, 1]
        })
        
        simulator.execute(df)
        
        # Verify state changed
        assert simulator.cash != simulator.initial_cash
        assert len(simulator.trades) > 0
        
        # Reset
        simulator.reset()
        
        # Verify state reset
        assert simulator.cash == simulator.initial_cash
        assert simulator.position == 0
        assert len(simulator.trades) == 0

    def test_missing_signals_column(self, simulator):
        """Test error handling when signals column is missing."""
        df = pl.DataFrame({
            'timestamp': pd.date_range(start="2024-01-01", end="2024-01-03", freq="D"),
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000000] * 3
            # No 'signal' column
        })
        
        with pytest.raises(ValueError, match="must contain 'signal' column"):
            simulator.execute(df)

    def test_get_trades_df(self, simulator):
        """Test getting trades as DataFrame."""
        # Execute trades first
        dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
        df = pl.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5,
            'signal': [1, -1, 1, -1, 1]
        })
        
        simulator.execute(df)
        
        trades_df = simulator.get_trades_df()
        
        if len(simulator.trades) > 0:
            # Should have correct columns
            expected_cols = ['timestamp', 'side', 'price', 'quantity', 'commission', 'slippage', 'total_cost']
            assert all(col in trades_df.columns for col in expected_cols)
            
            # Should have same number of rows as trades
            assert len(trades_df) == len(simulator.trades)
        else:
            # Empty DataFrame should have correct schema
            expected_cols = ['timestamp', 'side', 'price', 'quantity', 'commission', 'slippage', 'total_cost']
            assert all(col in trades_df.columns for col in expected_cols)
            assert len(trades_df) == 0


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test trade dataclass creation."""
        trade = Trade(
            timestamp="2024-01-01",
            side="BUY",
            price=100.5,
            quantity=100,
            commission=5.0,
            slippage=0.5,
            total_cost=10055.0
        )
        
        assert trade.timestamp == "2024-01-01"
        assert trade.side == "BUY"
        assert trade.price == 100.5
        assert trade.quantity == 100
        assert trade.commission == 5.0
        assert trade.slippage == 0.5
        assert trade.total_cost == 10055.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])