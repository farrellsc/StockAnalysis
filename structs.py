from dataclasses import dataclass
from pandas import DataFrame
import json

from currency import BaseCurrency
from utils import INF
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolios")


@dataclass
class StockConfig:
    symbol: str
    normalize: bool = False
    data: Optional[DataFrame] = None


@dataclass
class MacroConfig:
    interest_rate: bool = False
    cpi: bool = False
    unemployment_rate: bool = False


@dataclass
class Trade:
    index: int
    symbol: str
    date: str  # datetime to perform the trade, e.g. '2025-01-01'
    volume: Optional[float] = None  # amount of stocks to buy (positive) / sell (negative)
    cash_amount: Optional[BaseCurrency] = None  # amount of cash to invest (positive) / divest (negative)
    percentage: Optional[float] = None  # percentage of portfolio to invest (positive) / divest (negative)
    desc: Optional[str] = None

    def __post_init__(self):
        """Validate input and convert cash_amount to volume if needed"""
        # Validate date is a business day
        trade_date = datetime.strptime(self.date, '%Y-%m-%d')
        if trade_date.weekday() >= 5:  # Saturday=5, Sunday=6
            raise ValueError(f"Trade date {self.date} is not a business day (weekday={trade_date.weekday()}). Please use a weekday (0-4).")

        # Validate that only one of volume, cash_amount, or percentage is specified
        specified_params = [self.volume is not None, self.cash_amount is not None, self.percentage is not None]
        if sum(specified_params) > 1:
            raise ValueError("Cannot specify more than one of volume, cash_amount, or percentage. Use only one.")

        if sum(specified_params) == 0:
            raise ValueError("Must specify one of: volume (number of shares), cash_amount (dollar amount), or percentage (% of portfolio).")

        # If cash_amount is specified, we need to convert it to volume
        # This requires price data, so we'll defer the conversion to when price is available
        if self.cash_amount is not None:
            self._original_cash_amount = self.cash_amount
            # Volume will be calculated when convert_cash_to_volume() is called

        # If percentage is specified, we need to convert it to volume
        # This requires both price data and portfolio value, so we'll defer the conversion
        if self.percentage is not None:
            self._original_percentage = self.percentage
            # Volume will be calculated when convert_percentage_to_volume() is called

    def convert_cash_to_volume(self, price: float):
        """Convert cash_amount to volume using the given price"""
        import math

        if self.cash_amount is not None and self.volume is None:
            if price <= 0:
                raise ValueError(f"Invalid price {price} for converting cash to volume")

            # Handle infinite cash amount
            if math.isinf(self.cash_amount):
                if self.cash_amount > 0:  # Positive infinity - buy all available
                    self.volume = INF  # Large number to represent unlimited buying power
                else:  # Negative infinity - sell all holdings
                    self.volume = -INF  # Large negative number to represent unlimited selling
            else:
                # Calculate volume from cash amount
                raw_volume = self.cash_amount / price

                # Handle NaN values
                if math.isnan(raw_volume):
                    raise ValueError(f"Cannot calculate volume: cash_amount={self.cash_amount}, price={price}")

                self.volume = raw_volume  # Keep as float for fractional shares

            # Clear cash_amount since we now have volume
            self.cash_amount = None

    def convert_percentage_to_volume(self, reference_volume: float):
        """Convert percentage to volume by taking the specified percentage of a reference volume"""
        import math

        if self.percentage is not None and self.volume is None:
            if reference_volume <= 0:
                raise ValueError(f"Invalid reference volume {reference_volume} for converting percentage to volume")

            # Calculate volume as percentage of reference volume
            raw_volume = (self.percentage / 100.0) * reference_volume
            self.volume = raw_volume  # Keep as float for fractional shares

            # Clear percentage since we now have volume
            self.percentage = None

@dataclass
class MockPortfolio:
    name: str
    trade_history: List[Trade]


@dataclass
class Holding:
    symbol: str
    volume: float


@dataclass
class Position:
    """Represents a portfolio position with all holdings and metrics"""
    holdings: Dict[str, Holding]  # symbol -> holding details
    cash: BaseCurrency
    date: str

    def merge(self, other: 'Position', name: str = "Combined") -> 'Position':
        """
        Merge this position with another position to create a combined overview.

        Args:
            other: Another Position object to merge with
            name: Name for the combined position summary

        Returns:
            New Position object with combined metrics
        """
        # Merge holdings - combine holdings with same symbol
        merged_holdings = {}

        # Add holdings from self
        for symbol, holding in self.holdings.items():
            merged_holdings[symbol] = holding.copy()

        # Add or merge holdings from other
        for symbol, holding in other.holdings.items():
            if symbol in merged_holdings:
                # Combine holdings for the same symbol
                existing = merged_holdings[symbol]
                merged_holdings[symbol] = {
                    'net_amount': existing.get('net_amount', 0) + holding.get('net_amount', 0),
                    'total_invested': existing.get('total_invested', 0) + holding.get('total_invested', 0),
                    'actual_shares': existing.get('actual_shares', 0) + holding.get('actual_shares', 0),
                    'current_value': existing.get('current_value', 0) + holding.get('current_value', 0),
                    'unrealized_pnl': (existing.get('unrealized_pnl') or 0) + (holding.get('unrealized_pnl') or 0),
                    'trades_count': existing.get('trades_count', 0) + holding.get('trades_count', 0),
                    'cash_percentage': 0,  # Will be recalculated if needed
                    # Use the first available price data
                    'avg_price': existing.get('avg_price') or holding.get('avg_price'),
                    'current_price': existing.get('current_price') or holding.get('current_price'),
                    'price_change_ratio': existing.get('price_change_ratio') or holding.get('price_change_ratio'),
                    'portfolio_portion': 0  # Will be recalculated
                }
            else:
                merged_holdings[symbol] = holding.copy()

        # Combine financial metrics
        combined_cash = self.cash + other.cash
        combined_total_invested = self.total_invested + other.total_invested
        combined_total_divested = self.total_divested + other.total_divested
        combined_total_portfolio_value = self.total_portfolio_value + other.total_portfolio_value
        combined_total_value = self.total_value + other.total_value
        combined_total_unrealized_pnl = self.total_unrealized_pnl + other.total_unrealized_pnl
        combined_cumulative_invested = self.cumulative_invested + other.cumulative_invested
        combined_total_trades = self.total_trades + other.total_trades

        # Calculate combined return rate
        combined_total_return_rate = 0
        if combined_total_invested > 0:
            combined_total_return_rate = combined_total_unrealized_pnl / combined_total_invested

        # Recalculate portfolio portions for merged holdings
        if combined_total_portfolio_value > 0:
            for symbol, holding in merged_holdings.items():
                current_value = holding.get('current_value', 0)
                holding['portfolio_portion'] = current_value / combined_total_portfolio_value
        else:
            # If no portfolio value, set all portions to 0
            for symbol, holding in merged_holdings.items():
                holding['portfolio_portion'] = 0

        # Create combined summary
        holdings_count = len([h for h in merged_holdings.values() if h.get('net_amount', 0) > 0])
        combined_summary = f"{name}: ${combined_cash:,.2f} cash, {holdings_count} positions, ${combined_total_portfolio_value:,.2f} invested"

        return Position(
            holdings=merged_holdings,
            cash=combined_cash,
            total_invested=combined_total_invested,
            total_divested=combined_total_divested,
            total_portfolio_value=combined_total_portfolio_value,
            total_value=combined_total_value,
            total_unrealized_pnl=combined_total_unrealized_pnl,
            total_return_rate=combined_total_return_rate,
            cumulative_invested=combined_cumulative_invested,
            total_trades=combined_total_trades,
            summary=combined_summary
        )

    def pretty_print(self, name: str = "Position", original_cash: float = None):
        """Print position data in a pretty format."""
        print(f"\n📊 {name}")
        print("=" * 100)

        # Cash position
        print(f"💰 Cash Available: ${self.cash:,.2f}")

        # Holdings with detailed metrics
        if self.holdings:
            print(f"\n📈 Current Positions:")
            print("-" * 120)
            print(f"{'Symbol':<8} {'Shares':<10} {'Invested':<12} {'Avg Price':<10} {'Curr Price':<10} {'Change':<8} {'Curr Value':<12} {'P&L':<12} {'Portion':<8} {'Cash %':<8}")
            print("-" * 128)

            total_pnl = 0
            for symbol, holding in self.holdings.items():
                actual_shares = holding.get('actual_shares', 0)
                invested = holding['total_invested']
                avg_price = holding['avg_price']
                current_price = holding['current_price']
                price_change = holding['price_change_ratio']
                current_value = holding['current_value']
                unrealized_pnl = holding['unrealized_pnl']
                portion = holding.get('portfolio_portion', 0)
                cash_percentage = holding.get('cash_percentage', 0)

                # Format values
                shares_str = f"{actual_shares:,.0f}" if actual_shares > 0 else "-"
                invested_str = f"${invested:,.0f}" if invested > 0 else "-"
                avg_price_str = f"${avg_price:.2f}" if avg_price else "-"
                curr_price_str = f"${current_price:.2f}" if current_price else "-"
                change_str = f"{price_change:+.1%}" if price_change is not None else "-"
                curr_value_str = f"${current_value:,.0f}" if current_value > 0 else "-"
                pnl_str = f"${unrealized_pnl:+,.0f}" if unrealized_pnl is not None else "-"
                portion_str = f"{portion:.1%}" if portion > 0 else "-"
                cash_pct_str = f"{cash_percentage:.1%}" if cash_percentage > 0 else "-"

                # Color coding for P&L (simplified text indicators)
                if unrealized_pnl is not None:
                    if unrealized_pnl > 0:
                        pnl_str = f"+${unrealized_pnl:,.0f} ✓"
                        total_pnl += unrealized_pnl
                    elif unrealized_pnl < 0:
                        pnl_str = f"-${abs(unrealized_pnl):,.0f} ✗"
                        total_pnl += unrealized_pnl
                    else:
                        pnl_str = "$0 ="

                print(f"{symbol:<8} {shares_str:<10} {invested_str:<12} {avg_price_str:<10} {curr_price_str:<10} {change_str:<8} {curr_value_str:<12} {pnl_str:<12} {portion_str:<8} {cash_pct_str:<8}")

            # Add TOTAL row as part of the holdings table
            print("-" * 128)

            # Summary totals
            total_shares = sum(holding.get('actual_shares', 0) for holding in self.holdings.values())

            # Format TOTAL row values
            total_shares_str = f"{total_shares:,.0f}" if total_shares > 0 else "-"
            total_invested_str = f"${self.total_invested:,.0f}" if self.total_invested > 0 else "-"
            total_curr_value_str = f"${self.total_portfolio_value:,.0f}" if self.total_portfolio_value > 0 else "-"

            # Format total P&L
            if total_pnl != 0:
                pnl_indicator = "✓" if total_pnl > 0 else "✗"
                total_pnl_str = f"${total_pnl:+,.0f} {pnl_indicator}"
            else:
                total_pnl_str = "$0 ="

            # Calculate total cash percentage used
            if original_cash and original_cash > 0 and self.total_invested > 0:
                total_cash_pct = self.total_invested / original_cash
                total_cash_pct_str = f"{total_cash_pct:.1%}"
            else:
                total_cash_pct_str = "-"

            print(f"{'TOTAL':<8} {total_shares_str:<10} {total_invested_str:<12} {'-':<10} {'-':<10} {'-':<8} {total_curr_value_str:<12} {total_pnl_str:<12} {'100.0%':<8} {total_cash_pct_str:<8}")
            print("-" * 128)

        else:
            print(f"\n📈 Current Positions: None")

        # Total Investment Metrics Section
        print(f"\n📊 Total Investment Metrics:")
        print("-" * 50)

        print(f"  💰 Total Cash Available:     ${self.cash:>12,.2f}")
        print(f"  📈 Current Invested Amount:  ${self.total_invested:>12,.2f}")
        print(f"  📉 Total Divested Amount:    ${self.total_divested:>12,.2f}")
        print(f"  💎 Current Portfolio Value:  ${self.total_portfolio_value:>12,.2f}")
        print(f"  🏦 Total Portfolio + Cash:   ${self.total_value:>12,.2f}")
        print(f"  📊 Cumulative Invested:      ${self.cumulative_invested:>12,.2f}")
        print("-" * 50)

        # Performance metrics
        pnl_indicator = "✓" if self.total_unrealized_pnl > 0 else "✗" if self.total_unrealized_pnl < 0 else "="
        return_indicator = "✓" if self.total_return_rate > 0 else "✗" if self.total_return_rate < 0 else "="

        print(f"  💹 Total Unrealized P&L:     ${self.total_unrealized_pnl:>+12,.2f} {pnl_indicator}")
        print(f"  📋 Total Return Rate:        {self.total_return_rate:>+12.2%} {return_indicator}")
        print(f"  🔄 Total Trades Executed:    {self.total_trades:>16,d}")

        if self.total_invested > 0:
            investment_efficiency = (self.total_portfolio_value / self.total_invested - 1) * 100
            efficiency_indicator = "✓" if investment_efficiency > 0 else "✗" if investment_efficiency < 0 else "="
            print(f"  ⚡ Investment Efficiency:     {investment_efficiency:>+12.1f}% {efficiency_indicator}")

        print("=" * 100)


@dataclass
class Portfolio:
    name: str
    cash: BaseCurrency
    percent: float
    portfolio: Dict[str, float]  # symbol -> allocation percentage

    def __post_init__(self):
        # Validate business days
        for record in self.records:
            if "time" in record:
                try:
                    timestamp = datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S")
                    # Check if it's a weekend (Saturday=5, Sunday=6)
                    assert timestamp.weekday() < 5, f"Timestamp {record['time']} is not a business day (weekday: {timestamp.weekday()})"
                except ValueError as e:
                    raise ValueError(f"Invalid timestamp format in record: {record['time']}. Expected format: YYYY-MM-DD HH:MM:SS") from e

    @classmethod
    def from_file(cls, filename: str, convert_currency: Optional[str] = None):
        """
        Load portfolio from file.

        Args:
            filename: Name of the portfolio file (without .json extension)
            convert_currency: Target currency for conversion (e.g., "USD").
                            If None, no conversion will be performed.
                            Currently supports CNY to USD conversion only.
        """
        content = json.load(open(PORTFOLIO_DIR + "/" + filename + ".json"))

        # Create portfolio instance with conversion flag set
        portfolio = cls(
            name=filename,
            cash=content.get("cash", 0),
            percent=content.get("percent", 0),
            portfolio=content.get("portfolio", {}),
        )

        return portfolio

    def to_file(self):
        content = {
            "cash": self.cash,
            "percent": self.percent,
            "portfolio": self.portfolio,
        }
        json.dump(content, open(PORTFOLIO_DIR + "/" + self.name + ".json", "w"), indent=2)