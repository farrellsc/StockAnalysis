from dataclasses import dataclass
from pandas import DataFrame
import json
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
    symbol: str
    date: str  # datetime to perform the trade, e.g. '2025-01-01'
    volume: Optional[int] = None  # amount of stocks to buy (positive) / sell (negative)
    cash_amount: Optional[float] = None  # amount of cash to invest (positive) / divest (negative)
    percentage: Optional[float] = None  # percentage of portfolio to invest (positive) / divest (negative)

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

                self.volume = int(raw_volume)  # Round down to whole shares

            # Clear cash_amount since we now have volume
            self.cash_amount = None

    def convert_percentage_to_volume(self, price: float, portfolio_value: float):
        """Convert percentage to volume using the given price and portfolio value"""
        import math

        if self.percentage is not None and self.volume is None:
            if price <= 0:
                raise ValueError(f"Invalid price {price} for converting percentage to volume")

            if portfolio_value <= 0:
                raise ValueError(f"Invalid portfolio value {portfolio_value} for converting percentage to volume")

            # Calculate cash amount from percentage
            cash_amount = (self.percentage / 100.0) * portfolio_value

            # Handle infinite percentage
            if math.isinf(self.percentage):
                if self.percentage > 0:  # Positive infinity
                    self.volume = INF
                else:  # Negative infinity
                    self.volume = -INF
            else:
                # Calculate volume from cash amount
                raw_volume = cash_amount / price

                self.volume = int(raw_volume)  # Round down to whole shares

            # Clear percentage since we now have volume
            self.percentage = None

@dataclass
class MockPortfolio:
    name: str
    trade_history: List[Trade]


@dataclass
class Position:
    """Represents a portfolio position with all holdings and metrics"""
    holdings: Dict[str, Dict]  # symbol -> holding details
    cash: float
    total_invested: float
    total_divested: float
    total_portfolio_value: float
    total_value: float  # cash + portfolio value
    total_unrealized_pnl: float
    total_return_rate: float
    cumulative_invested: float
    total_trades: int
    summary: str

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
    cash: float
    percent: float
    portfolio: Dict[str, float]  # symbol -> allocation percentage
    records: List[Dict]  # list of timestamped holdings
    currency: str = "USD"  # Portfolio currency (USD, CNY, etc.)
    _conversion_skipped: bool = False  # Internal flag to control conversion

    def __post_init__(self):
        """Validate that all timestamps in records are business days and convert Chinese portfolio amounts"""
        # Currency conversion for Chinese portfolios (unless skipped)
        if not self._conversion_skipped:
            if self.currency == "CNY" and self._is_chinese_portfolio():
                self._convert_chinese_amounts_to_usd()

        # Validate business days
        for record in self.records:
            if "time" in record:
                try:
                    timestamp = datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S")
                    # Check if it's a weekend (Saturday=5, Sunday=6)
                    assert timestamp.weekday() < 5, f"Timestamp {record['time']} is not a business day (weekday: {timestamp.weekday()})"
                except ValueError as e:
                    raise ValueError(f"Invalid timestamp format in record: {record['time']}. Expected format: YYYY-MM-DD HH:MM:SS") from e

    def _is_chinese_portfolio(self) -> bool:
        """Check if this is a Chinese portfolio based on symbols in portfolio allocation or records"""
        # Check portfolio allocation symbols
        for symbol in self.portfolio.keys():
            if symbol.startswith('SH') or symbol.startswith('SZ'):
                return True

        # Check symbols in trade records
        for record in self.records:
            for key in record.keys():
                if key != "time" and (key.startswith('SH') or key.startswith('SZ')):
                    return True

        return False

    def _get_cny_to_usd_rate(self, date: str = None) -> float:
        """Get CNY to USD exchange rate for a specific date. Falls back to approximate rate if no data."""
        # Approximate CNY/USD rate (you might want to fetch real-time or historical data)
        # As of recent years, roughly 1 USD = 7.0-7.3 CNY, so 1 CNY = ~0.14 USD

        if date:
            # Try to get historical rate for the specific date
            # This is a simplified implementation - in production you'd want to fetch real rates
            try:
                # You could integrate with forex APIs here
                # For now, use approximate historical rates
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                year = date_obj.year

                # Approximate historical rates
                if year >= 2024:
                    return 0.138  # ~1/7.25
                elif year >= 2022:
                    return 0.148  # ~1/6.75
                elif year >= 2020:
                    return 0.143  # ~1/7.0
                else:
                    return 0.145  # ~1/6.9
            except:
                pass

        # Default current approximate rate
        return 0.138  # ~1/7.25 USD per CNY

    def _convert_chinese_amounts_to_usd(self):
        """Convert Chinese portfolio cash and trade amounts from CNY to USD"""
        if self.currency == "CNY":
            # Convert cash amount using current rate
            current_rate = self._get_cny_to_usd_rate()
            original_cash = self.cash
            self.cash = self.cash * current_rate

            print(f"🔄 Converting Chinese portfolio '{self.name}' from CNY to USD:")
            print(f"   Cash: ¥{original_cash:,.2f} → ${self.cash:,.2f} (rate: {current_rate:.4f})")

            # Convert trade record amounts using historical rates
            converted_records = 0
            for record in self.records:
                if "time" in record:
                    record_date = datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
                    rate = self._get_cny_to_usd_rate(record_date)

                    for symbol, amount in record.items():
                        if symbol != "time" and isinstance(amount, (int, float)):
                            original_amount = amount
                            record[symbol] = amount * rate
                            converted_records += 1

            print(f"   Converted {converted_records} trade record amounts using historical rates")

            # Update currency designation
            self.currency = "USD"

    def convert_currency(self, target_currency: str):
        """
        Manually convert portfolio to target currency.

        Args:
            target_currency: Target currency code (e.g., "USD")
        """
        if target_currency == "USD" and self.currency == "CNY":
            self._convert_chinese_amounts_to_usd()
            self._conversion_skipped = False
            print(f"✅ Portfolio converted from CNY to USD")
        elif target_currency == self.currency:
            print(f"Portfolio is already in {self.currency}, no conversion needed")
        else:
            supported_conversions = "CNY -> USD"
            raise ValueError(f"Unsupported currency conversion: {self.currency} -> {target_currency}. "
                           f"Currently supported: {supported_conversions}")

    def convert_to_usd(self):
        """Convenience method to convert to USD (backward compatibility)"""
        self.convert_currency("USD")

    @property
    def date(self) -> str:
        """Get the most recent date from records"""
        if not self.records:
            return datetime.now().strftime("%Y-%m-%d")

        most_recent_time = max(datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S") for record in self.records)
        return most_recent_time.strftime("%Y-%m-%d")

    @property
    def compositions(self) -> Dict[str, float]:
        """Get the portfolio allocations for backward compatibility"""
        return self.portfolio

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

        # Detect original currency from file or auto-detect
        original_currency = content.get("currency", "USD")

        # Determine if conversion should be skipped before creating instance
        conversion_skipped = convert_currency is None

        # Create portfolio instance with conversion flag set
        portfolio = cls(
            name=filename,
            cash=content.get("cash", 0),
            percent=content.get("percent", 0),
            portfolio=content.get("portfolio", {}),
            records=content.get("records", []),
            currency=original_currency,
            _conversion_skipped=conversion_skipped,
        )

        return portfolio

    def to_file(self):
        content = {
            "cash": self.cash,
            "percent": self.percent,
            "portfolio": self.portfolio,
            "records": self.records,
            "currency": self.currency
        }
        json.dump(content, open(PORTFOLIO_DIR + "/" + self.name + ".json", "w"), indent=2)

    def add_trade_record(self, symbol: str, trade_amount: float, timestamp: str = None):
        """Add a new trade record for a symbol (positive = buy, negative = sell)"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "time": timestamp,
            symbol: trade_amount
        }
        self.records.append(record)

    def get_current_position(self, backend=None, target_date: str = None, print_output: bool = True) -> Position:
        """
        Get current position by accumulating all records chronologically.

        Args:
            backend: Backend instance for price queries (if None, will create one)
            target_date (str): Date for price queries in 'YYYY-MM-DD' format (if None, uses today's date)
            print_output (bool): Whether to print formatted output. Defaults to True.

        Returns:
            Position object containing all portfolio holdings and metrics
        """
        if not self.records:
            result = Position(
                holdings={},
                cash=self.cash,
                total_invested=0,
                total_divested=0,
                total_portfolio_value=0,
                total_value=self.cash,
                total_unrealized_pnl=0,
                total_return_rate=0,
                cumulative_invested=0,
                total_trades=0,
                summary=f"Portfolio '{self.name}': ${self.cash:,.2f} cash, no holdings"
            )
        else:
            # Initialize Backend if not provided
            if backend is None:
                from backend import Backend
                from database import Database
                backend = Backend(database=Database(file_path=os.path.join(DATA_DIR, "stock_data.pkl")))

            # Determine target date for price queries
            if target_date is None:
                # Use today's date, but adjust to most recent business day if needed
                target_date_dt = datetime.now()
                # If today is weekend, go back to Friday
                if target_date_dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    days_back = target_date_dt.weekday() - 4  # Go back to Friday (4)
                    target_date_dt = target_date_dt - timedelta(days=days_back)
                target_date = target_date_dt.strftime("%Y-%m-%d")

                # Check if target_date exists in database, if not fall back to latest available date
                try:
                    # Try to get any symbol's data for target_date to check if date exists
                    test_symbols = list(set(symbol for record in self.records for symbol in record.keys() if symbol != "time"))
                    date_exists = False

                    if test_symbols:
                        test_data = backend.get_daily_price(test_symbols[0], target_date, target_date)
                        date_exists = test_data is not None and len(test_data) > 0

                    # If target_date doesn't exist in database, fall back to latest available date
                    if not date_exists:
                        latest_date = backend.database.get_latest_date()
                        if latest_date:
                            target_date = latest_date
                except Exception:
                    # If there's any error checking, stick with the original target_date
                    pass

            # Sort records chronologically for processing
            sorted_records = sorted(self.records, key=lambda x: datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S"))

            # Accumulate trade records and calculate shares for each trade
            position_details = {}
            for record in sorted_records:
                record_date = datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

                for symbol, trade_amount in record.items():
                    if symbol != "time":
                        if symbol not in position_details:
                            position_details[symbol] = {
                                'total_invested': 0,
                                'net_amount': 0,
                                'trades': [],
                                'total_shares': 0,
                                'weighted_avg_price': 0
                            }

                        position_details[symbol]['net_amount'] += trade_amount

                        # Get historical price for this trade date
                        try:
                            price_data = backend.get_daily_price(symbol, record_date, record_date)

                            # If no data for trade date, try fallback dates
                            if price_data is None or len(price_data) == 0:
                                fallback_date = datetime.strptime(record_date, "%Y-%m-%d")
                                for i in range(1, 6):  # Try up to 5 days forward and backward
                                    # Try going forward first
                                    forward_date = fallback_date + timedelta(days=i)
                                    if forward_date.weekday() < 5:  # Business day
                                        forward_date_str = forward_date.strftime("%Y-%m-%d")
                                        price_data = backend.get_daily_price(symbol, forward_date_str, forward_date_str)
                                        if price_data is not None and len(price_data) > 0:
                                            break

                                    # Try going backward
                                    backward_date = fallback_date - timedelta(days=i)
                                    if backward_date.weekday() < 5:  # Business day
                                        backward_date_str = backward_date.strftime("%Y-%m-%d")
                                        price_data = backend.get_daily_price(symbol, backward_date_str, backward_date_str)
                                        if price_data is not None and len(price_data) > 0:
                                            break

                            if price_data is not None and len(price_data) > 0:
                                trade_price = price_data['Close'].iloc[0]
                                shares_traded = trade_amount / trade_price if trade_price > 0 else 0

                                position_details[symbol]['trades'].append({
                                    'amount': trade_amount,
                                    'timestamp': record['time'],
                                    'price': trade_price,
                                    'shares': shares_traded
                                })

                                # Track total shares accumulated (including buys and sells)
                                position_details[symbol]['total_shares'] += shares_traded

                                # Only count positive trades for total invested calculation
                                if trade_amount > 0:
                                    position_details[symbol]['total_invested'] += trade_amount
                            else:
                                # No price data available for this date - use amount as placeholder
                                print(f"Warning: No price data for {symbol} on {record_date}, using amount as shares")
                                position_details[symbol]['trades'].append({
                                    'amount': trade_amount,
                                    'timestamp': record['time'],
                                    'price': None,
                                    'shares': trade_amount  # Use amount as fallback
                                })
                                position_details[symbol]['total_shares'] += trade_amount
                                if trade_amount > 0:
                                    position_details[symbol]['total_invested'] += trade_amount
                        except Exception as e:
                            # Error getting price data
                            position_details[symbol]['trades'].append({
                                'amount': trade_amount,
                                'timestamp': record['time'],
                                'price': None,
                                'shares': None,
                                'error': str(e)
                            })
                            if trade_amount > 0:
                                position_details[symbol]['total_invested'] += trade_amount

            # Calculate enhanced metrics for each position
            current_holdings = {}
            total_portfolio_value = 0
            total_invested = 0
            total_divested = 0

            for symbol, details in position_details.items():
                net_amount = details['net_amount']
                if net_amount == 0:  # Skip positions with zero net amount
                    continue

                # Calculate weighted average price (only for positive investments)
                avg_price = None
                total_shares_bought = sum(trade['shares'] for trade in details['trades']
                                        if trade['shares'] is not None and trade['shares'] > 0)
                if details['total_invested'] > 0 and total_shares_bought > 0:
                    avg_price = details['total_invested'] / total_shares_bought

                # Get current price
                current_price = None
                current_value = 0

                # Always set actual_shares from details, regardless of price availability
                actual_shares = details['total_shares']  # This already includes buys (+) and sells (-)

                try:
                    price_data = backend.get_daily_price(symbol, target_date, target_date)

                    # If no data for target_date, try to go back up to 10 business days
                    if price_data is None or len(price_data) == 0:
                        fallback_date = datetime.strptime(target_date, "%Y-%m-%d")
                        for i in range(1, 11):  # Try up to 10 days back
                            fallback_date = fallback_date - timedelta(days=1)
                            # Skip weekends
                            if fallback_date.weekday() < 5:  # Monday=0 to Friday=4
                                fallback_date_str = fallback_date.strftime("%Y-%m-%d")
                                price_data = backend.get_daily_price(symbol, fallback_date_str, fallback_date_str)
                                if price_data is not None and len(price_data) > 0:
                                    break

                    if price_data is not None and len(price_data) > 0:
                        current_price = price_data['Close'].iloc[0]
                        current_value = actual_shares * current_price if actual_shares > 0 else 0
                except Exception as e:
                    # Add some debugging info
                    print(f"Error getting price for {symbol} on {target_date}: {e}")
                    current_price = None
                    current_value = 0

                # Calculate price change ratio
                price_change_ratio = None
                if avg_price and current_price:
                    price_change_ratio = (current_price - avg_price) / avg_price

                # Calculate percentage of total cash used for this holding
                cash_percentage = (details['total_invested'] / self.cash) if self.cash > 0 and details['total_invested'] > 0 else 0

                current_holdings[symbol] = {
                    'net_amount': net_amount,
                    'total_invested': details['total_invested'],
                    'actual_shares': actual_shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'current_value': current_value,
                    'price_change_ratio': price_change_ratio,
                    'unrealized_pnl': current_value - details['total_invested'] if current_value > 0 and details['total_invested'] > 0 else None,
                    'trades_count': len(details['trades']),
                    'cash_percentage': cash_percentage
                }

                # Accumulate totals
                if net_amount > 0:
                    total_invested += net_amount
                    total_portfolio_value += current_value if current_value > 0 else net_amount
                else:
                    total_divested += abs(net_amount)

            # Calculate portfolio portions
            if total_portfolio_value > 0:
                for symbol, holding in current_holdings.items():
                    # Use the same logic as total_portfolio_value calculation
                    current_value = holding.get('current_value', 0)
                    net_amount = holding.get('net_amount', 0)
                    if net_amount > 0:
                        value_for_portion = current_value if current_value > 0 else net_amount
                        holding['portfolio_portion'] = value_for_portion / total_portfolio_value
                    else:
                        holding['portfolio_portion'] = 0
            else:
                # If no portfolio value, set all portions to 0
                for symbol, holding in current_holdings.items():
                    holding['portfolio_portion'] = 0

            # Calculate remaining cash
            remaining_cash = self.cash + total_divested - total_invested

            # Calculate total investment metrics
            total_unrealized_pnl = sum(holding['unrealized_pnl'] for holding in current_holdings.values()
                                     if holding['unrealized_pnl'] is not None)
            total_return_rate = (total_unrealized_pnl / total_invested) if total_invested > 0 else 0
            total_portfolio_with_cash = remaining_cash + total_portfolio_value

            # Calculate cumulative invested amount (all money put into the market over time)
            cumulative_invested = sum(trade['amount'] for details in position_details.values()
                                    for trade in details['trades'] if trade['amount'] > 0)

            # Calculate total trades count
            total_trades = sum(len(details['trades']) for details in position_details.values())

            # Create summary
            holdings_summary = []
            for symbol, holding in current_holdings.items():
                net_amount = holding['net_amount']
                if net_amount > 0:
                    pnl_str = ""
                    if holding['unrealized_pnl'] is not None:
                        pnl_str = f" (P&L: ${holding['unrealized_pnl']:+,.2f})"
                    holdings_summary.append(f"{symbol}: ${net_amount:,.2f}{pnl_str}")
                else:
                    holdings_summary.append(f"{symbol}: -${abs(net_amount):,.2f}")

            if holdings_summary:
                holdings_text = ", ".join(holdings_summary)
                summary = f"Portfolio '{self.name}': ${remaining_cash:,.2f} cash, {holdings_text}"
            else:
                summary = f"Portfolio '{self.name}': ${remaining_cash:,.2f} cash, no positions"

            result = Position(
                holdings=current_holdings,  # Detailed position information per symbol
                cash=remaining_cash,
                total_invested=total_invested,  # Total current long positions
                total_divested=total_divested,  # Total short positions
                total_portfolio_value=total_portfolio_value,  # Current portfolio value
                total_value=remaining_cash + total_portfolio_value,  # Cash + portfolio value
                total_unrealized_pnl=total_unrealized_pnl,  # Total unrealized profit/loss
                total_return_rate=total_return_rate,  # Overall return rate
                cumulative_invested=cumulative_invested,  # Total money put into market over time
                total_trades=total_trades,  # Total number of trades executed
                summary=summary
            )

        # Print pretty formatted output if requested
        if print_output:
            self._print_position(result)

        return result

    def pretty_print(self):
        """Print portfolio-specific information."""
        print(f"\n🎯 Portfolio Configuration: {self.name}")
        print("=" * 60)

        # Portfolio allocation (from configuration)
        if self.portfolio:
            print(f"\n🎯 Target Allocation:")
            print("-" * 30)
            for symbol, allocation in self.portfolio.items():
                print(f"  {symbol:10s}: {allocation:>7.1%}")

        # Investment parameters
        print(f"\n⚙️  Investment Parameters:")
        print("-" * 30)
        currency_symbol = "$" if self.currency == "USD" else "¥" if self.currency == "CNY" else self.currency
        print(f"  Total Cash: {currency_symbol}{self.cash:>12,.2f} ({self.currency})")
        print(f"  Investment %: {self.percent:>6.1%}")
        print(f"  Target Amount: {currency_symbol}{self.cash * self.percent:>8,.2f}")

        # Last update
        if self.records:
            latest_time = max(datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S") for record in self.records)
            print(f"\n🕒 Last Updated: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("=" * 60)

    def _print_position(self, position_data: Position):
        """Print combined portfolio and position data."""
        # Print position data
        position_data.pretty_print(f"Portfolio Position: {self.name}", original_cash=self.cash)

        # Print portfolio-specific configuration
        self.pretty_print()

    def get_next_trade(self, backend=None, target_date: str = None, print_output: bool = True) -> Dict:
        """
        Calculate the next trade to balance portfolio towards target allocation.

        Args:
            backend: Backend instance for price queries (if None, will create one)
            target_date (str): Date for price queries in 'YYYY-MM-DD' format (if None, uses today's date)
            print_output (bool): Whether to print formatted output. Defaults to True.

        Returns:
            Dict with structure:
            {
                'investment_amount': float,  # Amount to invest
                'trades': {symbol: amount_to_buy},  # Recommended trades
                'current_allocations': {symbol: current_percentage},
                'target_allocations': {symbol: target_percentage},
                'allocation_gaps': {symbol: gap_percentage},
                'rationale': str  # Explanation of the trade recommendation
            }
        """
        # Get current position data
        current_position = self.get_current_position(backend=backend, target_date=target_date, print_output=False)

        # Calculate investment amount (percentage of cash)
        investment_amount = self.cash * self.percent

        # Get current holdings and their allocations
        current_holdings = current_position.holdings
        total_portfolio_value = current_position.total_portfolio_value

        # Calculate current allocations as percentages
        current_allocations = {}
        for symbol in self.portfolio.keys():
            if symbol in current_holdings and total_portfolio_value > 0:
                current_value = current_holdings[symbol].get('current_value', 0)
                current_allocations[symbol] = current_value / total_portfolio_value
            else:
                current_allocations[symbol] = 0.0

        # Calculate allocation gaps (target - current)
        allocation_gaps = {}
        for symbol, target_pct in self.portfolio.items():
            current_pct = current_allocations.get(symbol, 0.0)
            allocation_gaps[symbol] = target_pct - current_pct

        # Calculate total portfolio value after investment
        future_portfolio_value = total_portfolio_value + investment_amount

        # Calculate required amounts to reach target allocations
        target_amounts = {}
        for symbol, target_pct in self.portfolio.items():
            target_amount = future_portfolio_value * target_pct
            current_value = current_holdings.get(symbol, {}).get('current_value', 0)
            required_investment = max(0, target_amount - current_value)
            target_amounts[symbol] = required_investment

        # Normalize investments to match available investment amount
        total_required = sum(target_amounts.values())

        recommended_trades = {}
        if total_required > 0:
            # Scale down proportionally if we need more than we have
            scale_factor = min(1.0, investment_amount / total_required)
            for symbol, required in target_amounts.items():
                if required > 0:
                    recommended_trades[symbol] = required * scale_factor
        else:
            # If all targets are met, distribute equally among underweight positions
            underweight_symbols = [s for s, gap in allocation_gaps.items() if gap > 0]
            if underweight_symbols:
                equal_amount = investment_amount / len(underweight_symbols)
                for symbol in underweight_symbols:
                    recommended_trades[symbol] = equal_amount

        # Generate rationale
        rationale_parts = []
        rationale_parts.append(f"Investment amount: ${investment_amount:,.2f} ({self.percent:.1%} of ${self.cash:,.2f} cash)")

        # Sort gaps by magnitude for reporting
        sorted_gaps = sorted(allocation_gaps.items(), key=lambda x: abs(x[1]), reverse=True)

        overweight = [(s, g) for s, g in sorted_gaps if g < -0.01]  # More than 1% overweight
        underweight = [(s, g) for s, g in sorted_gaps if g > 0.01]   # More than 1% underweight

        if overweight:
            rationale_parts.append(f"Overweight positions: {', '.join([f'{s} ({g:+.1%})' for s, g in overweight])}")

        if underweight:
            rationale_parts.append(f"Underweight positions: {', '.join([f'{s} ({g:+.1%})' for s, g in underweight])}")

        if recommended_trades:
            trade_summary = [f"{symbol}: ${amount:,.2f}" for symbol, amount in recommended_trades.items() if amount > 0]
            rationale_parts.append(f"Recommended trades: {', '.join(trade_summary)}")
        else:
            rationale_parts.append("No trades recommended - portfolio is balanced")

        result = {
            'investment_amount': investment_amount,
            'trades': recommended_trades,
            'current_allocations': current_allocations,
            'target_allocations': self.portfolio.copy(),
            'allocation_gaps': allocation_gaps,
            'rationale': '. '.join(rationale_parts)
        }

        # Print formatted output if requested
        if print_output:
            self._print_next_trade(result)

        return result

    def _print_next_trade(self, trade_data: Dict):
        """Print next trade recommendation in a pretty format."""
        print(f"\n🎯 Next Trade Recommendation: {self.name}")
        print("=" * 80)

        investment_amount = trade_data['investment_amount']
        trades = trade_data['trades']
        current_allocations = trade_data['current_allocations']
        target_allocations = trade_data['target_allocations']
        allocation_gaps = trade_data['allocation_gaps']

        print(f"💰 Investment Amount: ${investment_amount:,.2f}")
        print(f"📊 Cash Percentage Used: {self.percent:.1%}")

        # Allocation comparison table
        print(f"\n📈 Allocation Analysis:")
        print("-" * 70)
        print(f"{'Symbol':<8} {'Current':<10} {'Target':<10} {'Gap':<10} {'Recommended':<12}")
        print("-" * 70)

        for symbol in target_allocations.keys():
            current_pct = current_allocations.get(symbol, 0)
            target_pct = target_allocations[symbol]
            gap = allocation_gaps[symbol]
            recommended = trades.get(symbol, 0)

            current_str = f"{current_pct:.1%}"
            target_str = f"{target_pct:.1%}"
            gap_str = f"{gap:+.1%}"
            recommended_str = f"${recommended:,.0f}" if recommended > 0 else "-"

            # Add indicators
            if abs(gap) > 0.01:  # More than 1% gap
                if gap > 0:
                    gap_str += " ⬇"  # Underweight
                else:
                    gap_str += " ⬆"  # Overweight
            else:
                gap_str += " ✓"  # Balanced

            print(f"{symbol:<8} {current_str:<10} {target_str:<10} {gap_str:<10} {recommended_str:<12}")

        print("-" * 70)

        # Summary
        total_recommended = sum(trades.values())
        print(f"{'TOTAL':<8} {'':<10} {'100.0%':<10} {'':<10} ${total_recommended:,.0f}")

        # Rationale
        print(f"\n💡 Rationale:")
        print(f"   {trade_data['rationale']}")

        print("=" * 80)