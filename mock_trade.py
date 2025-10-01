from dataclasses import dataclass
from typing import List, Dict, Optional
from backend import Backend
from pandas import DataFrame
import pandas as pd


@dataclass
class Trade:
    symbol: str
    volume: int  # amount of stocks to buy (positive) / sell (negative)
    date: str  # datetime to perform the trade, e.g. '2025-01-01'


@dataclass
class MockTrade:
    backend: Backend
    trade_history: List[Trade]
    start_date: str
    end_date: str

    def __post_init__(self):
        self.symbol_data: Dict[str, DataFrame] = {}
        self.portfolio_df: Optional[DataFrame] = None
        self.holdings: Dict[str, int] = {}  # symbol -> shares held
        self.date_range: pd.DatetimeIndex = None
        self.current_trade_index: int = 0
        self.initial_investment: float = 0.0
        self._cached_trade_dates: Dict[int, pd.Timestamp] = {}  # Cache adjusted trade dates

    def mock(self) -> DataFrame:
        """Execute the mock trading simulation and return portfolio performance."""
        self.initialize_mock()

        # Process all trades in order
        while self.current_trade_index < len(self.trade_history):
            self.trade_one()
            self.current_trade_index += 1

        return self.portfolio_df

    def initialize_mock(self):
        """
        1. fetch all relevant stock prices across daterange from backend
        2. initialize portfolio data frames
        3. assume the investment amount == sum of stock values in the 1st trade date
        """
        # Get all unique symbols from trade history
        trade_symbols = set(trade.symbol for trade in self.trade_history)

        # Fetch price data for all symbols
        for symbol in trade_symbols:
            self.symbol_data[symbol] = self.backend.get_daily_price(symbol, self.start_date, self.end_date)

        # Create business day date range
        self.date_range = pd.bdate_range(start=self.start_date, end=self.end_date, freq='B')

        # Initialize portfolio DataFrame
        self.portfolio_df = pd.DataFrame(index=self.date_range)
        self.portfolio_df.index.name = 'date'
        self.portfolio_df['total_value'] = 0.0
        self.portfolio_df['cash'] = 0.0
        self.portfolio_df['portfolio_value'] = 0.0

        # Initialize holdings
        for symbol in trade_symbols:
            self.holdings[symbol] = 0

        # Calculate initial investment based on first trade date
        if self.trade_history:
            first_trade_date = pd.to_datetime(self.trade_history[0].date)
            if first_trade_date.weekday() >= 5:  # Weekend adjustment
                first_trade_date = first_trade_date + pd.Timedelta(days=7 - first_trade_date.weekday())

            # Calculate initial investment as sum of all initial trades on first date
            initial_value = 0.0
            for trade in self.trade_history:
                trade_date = pd.to_datetime(trade.date)
                if trade_date.weekday() >= 5:
                    trade_date = trade_date + pd.Timedelta(days=7 - trade_date.weekday())

                if trade_date == first_trade_date and trade.volume > 0:  # Only count buys
                    if trade.symbol in self.symbol_data:
                        prices = self.symbol_data[trade.symbol]['Close'].reindex(self.date_range, method='ffill')
                        if trade_date in prices.index:
                            price = prices.loc[trade_date]
                            initial_value += trade.volume * price

            self.initial_investment = initial_value
            print(f"Calculated initial investment: ${self.initial_investment:.2f}")

        # Set initial cash
        self.portfolio_df['cash'] = self.initial_investment
        self.portfolio_df['portfolio_value'] = self.initial_investment
        print(f"Portfolio initialized with ${self.initial_investment:.2f} cash")

        # Pre-cache adjusted trade dates for performance
        self._cache_trade_dates()

        # Initialize trade logging table
        self._init_trade_log()

    def _cache_trade_dates(self):
        """Pre-cache adjusted trade dates for performance optimization."""
        for i, trade in enumerate(self.trade_history):
            trade_date = pd.to_datetime(trade.date)
            if trade_date.weekday() >= 5:  # Weekend adjustment
                trade_date = trade_date + pd.Timedelta(days=7 - trade_date.weekday())
            self._cached_trade_dates[i] = trade_date

    def _init_trade_log(self):
        """Initialize trade logging table."""
        print("\n" + "="*80)
        print(f"{'TRADE LOG':<80}")
        print("="*80)
        print(f"{'#':<3} {'Date':<12} {'Action':<6} {'Symbol':<8} {'Volume':<8} {'Price':<10} {'Value':<12} {'Cash After':<12} {'Holdings':<15}")
        print("-"*80)

    def _calculate_holdings_at_date(self, symbol: str, target_date: pd.Timestamp, up_to_trade_index: int = None) -> int:
        """
        Calculate holdings for a specific symbol at a given date by replaying trades.
        This method applies clipping logic to get accurate holdings.

        Args:
            symbol: Symbol to calculate holdings for
            target_date: Date to calculate holdings at
            up_to_trade_index: Only process trades up to this index (exclusive)

        Returns:
            int: Holdings amount at the target date
        """
        holdings = 0
        max_index = up_to_trade_index if up_to_trade_index is not None else len(self.trade_history)

        for i in range(max_index):
            trade = self.trade_history[i]
            trade_date = self._cached_trade_dates[i]

            # Only count trades for same symbol that happened on or before target date
            if trade.symbol == symbol and trade_date <= target_date:
                actual_volume = trade.volume

                # Apply clipping for sell orders
                if trade.volume < 0:  # Sell order
                    # Calculate holdings just before this trade
                    holdings_before_trade = self._calculate_holdings_at_date(symbol, trade_date, i)
                    max_sellable = -holdings_before_trade  # Negative because volume is negative for sells

                    if trade.volume < max_sellable:  # Trying to sell more than we have
                        actual_volume = max_sellable

                holdings += actual_volume

        return holdings

    def _apply_clipping_logic(self, trade: Trade, trade_index: int, target_date: pd.Timestamp) -> int:
        """
        Apply clipping logic for sell orders that exceed available holdings.

        Args:
            trade: The trade to potentially clip
            trade_index: Index of the current trade (for replay calculation)
            target_date: Date up to which to calculate holdings

        Returns:
            int: The actual volume after clipping (same as original if no clipping needed)
        """
        actual_volume = trade.volume

        if trade.volume < 0:  # Sell order
            # Calculate holdings at the trade date by replaying all previous trades
            holdings_at_date = self._calculate_holdings_at_date(trade.symbol, target_date, trade_index)
            max_sellable = -holdings_at_date  # Negative because volume is negative for sells

            if trade.volume < max_sellable:  # Trying to sell more than we have
                actual_volume = max_sellable

        return actual_volume

    def trade_one(self):
        """
        1. if trade_history is not empty, pop the first one in the list and compute portfolio performance
        """
        if self.current_trade_index >= len(self.trade_history):
            return

        trade = self.trade_history[self.current_trade_index]

        # Use cached trade date (performance optimization)
        trade_date = self._cached_trade_dates[self.current_trade_index]

        # Skip if trade date is outside our range
        if trade_date not in self.date_range:
            return

        symbol = trade.symbol
        volume = trade.volume
        action = "BUY" if volume > 0 else "SELL"

        if symbol in self.symbol_data:
            # Get price data for this symbol
            prices = self.symbol_data[symbol]['Close'].reindex(self.date_range, method='ffill')

            if trade_date in prices.index:
                trade_price = prices.loc[trade_date]
                trade_value = volume * trade_price

                # Update cash (subtract for buys, add for sells)
                cash_change = -trade_value
                current_cash = self.portfolio_df.loc[trade_date, 'cash']

                # Check if we have enough cash for buy orders
                if volume > 0:  # Buy order
                    if current_cash + cash_change < 0:
                        max_affordable_volume = int(current_cash / trade_price)
                        raise ValueError(
                            f"Insufficient cash for trade on {trade_date.date()}: trying to buy {volume} shares of {symbol} "
                            f"at ${trade_price:.2f} = ${trade_value:.2f}, but only have ${current_cash:.2f} cash. "
                            f"Maximum affordable volume: {max_affordable_volume} shares "
                            f"(${max_affordable_volume * trade_price:.2f})"
                        )

                # Apply clipping logic using shared method
                actual_volume = self._apply_clipping_logic(trade, self.current_trade_index, trade_date)

                if actual_volume != volume:
                    # Update cash change and trade value to match actual trade
                    actual_trade_value = actual_volume * trade_price
                    cash_change = -actual_trade_value
                    trade_value = actual_trade_value

                    holdings_before_trade = 0
                    for i in range(self.current_trade_index):
                        prev_trade = self.trade_history[i]
                        prev_trade_date = self._cached_trade_dates[i]
                        if prev_trade.symbol == symbol and prev_trade_date <= trade_date:
                            holdings_before_trade += self._apply_clipping_logic(prev_trade, i, prev_trade_date)

                    print(f"WARNING: Sell order clipped - trying to sell {abs(volume)} {symbol} but only have {holdings_before_trade} shares")
                    print(f"         Adjusted to sell {abs(actual_volume)} shares instead")

                # Update holdings
                self.holdings[symbol] += actual_volume

                # Apply trade from trade_date onwards
                for date in self.date_range[self.date_range >= trade_date]:
                    self.portfolio_df.loc[date, 'cash'] += cash_change

                # Format table row logging
                new_cash = current_cash + cash_change
                holdings_str = f"{symbol}:{self.holdings[symbol]}"
                clipped_note = " (CLIPPED)" if actual_volume != volume else ""
                print(f"{self.current_trade_index+1:<3} {trade_date.strftime('%Y-%m-%d'):<12} {action:<6} {symbol:<8} {abs(actual_volume):<8} ${trade_price:<9.2f} ${abs(trade_value):<11.2f} ${new_cash:<11.2f} {holdings_str:<15}{clipped_note}")

        # Recalculate portfolio values (optimized version)
        self._update_portfolio_values()

    def _update_portfolio_values(self):
        """Update total portfolio values based on holdings at each date - optimized version."""
        # Pre-process and cache price data for all symbols (major optimization)
        cached_prices = {}
        for symbol in self.symbol_data:
            cached_prices[symbol] = self.symbol_data[symbol]['Close'].reindex(self.date_range, method='ffill')

        # Build holdings using vectorized operations
        holdings_df = pd.DataFrame(0, index=self.date_range, columns=list(self.holdings.keys()))

        # Process trades efficiently using cached dates
        for i in range(self.current_trade_index + 1):
            trade = self.trade_history[i]
            trade_date = self._cached_trade_dates[i]  # Use cached date

            if trade_date not in self.date_range:
                continue

            # Vectorized update - much faster than loop
            mask = holdings_df.index >= trade_date
            holdings_df.loc[mask, trade.symbol] += trade.volume

        # Vectorized portfolio value calculation
        total_values = pd.Series(0.0, index=self.date_range)

        for symbol in holdings_df.columns:
            if symbol in cached_prices:
                # Vectorized multiplication - much faster
                position_values = holdings_df[symbol] * cached_prices[symbol]
                total_values += position_values.fillna(0)

        # Batch update portfolio DataFrame
        self.portfolio_df['total_value'] = total_values
        self.portfolio_df['portfolio_value'] = self.portfolio_df['total_value'] + self.portfolio_df['cash']

        # Log current holdings after portfolio update
        print("\nCurrent Holdings Summary:")
        print("-" * 40)
        total_holding_value = 0.0
        for symbol, shares in self.holdings.items():
            if shares != 0 and symbol in cached_prices:
                current_price = cached_prices[symbol].iloc[-1]  # Latest price
                position_value = shares * current_price
                total_holding_value += position_value
                print(f"{symbol:<8}: {shares:>6} shares @ ${current_price:>8.2f} = ${position_value:>10.2f}")

        current_cash = self.portfolio_df['cash'].iloc[-1]
        total_portfolio = total_holding_value + current_cash

        print("-" * 40)
        print(f"{'Total':<8}: ${total_holding_value:>26.2f}")
        print(f"{'Cash':<8}: ${current_cash:>26.2f}")
        print(f"{'Portfolio':<8}: ${total_portfolio:>26.2f}")
        print("-" * 40)

    def get_holdings_for_date(self, date: str, show_report: bool = False) -> pd.DataFrame:
        """
        Get holdings and values for a specific date.

        Args:
            date (str): Date in 'YYYY-MM-DD' format
            show_report (bool): Whether to print a formatted report

        Returns:
            pd.DataFrame: Holdings data with columns ['symbol', 'shares', 'price', 'value']
        """
        target_date = pd.to_datetime(date)

        # Adjust to business day if needed
        if target_date.weekday() >= 5:
            target_date = target_date + pd.Timedelta(days=7 - target_date.weekday())
            if show_report:
                print(f"Adjusted weekend date {date} to business day {target_date.strftime('%Y-%m-%d')}")

        if target_date not in self.date_range:
            if show_report:
                print(f"Date {target_date.strftime('%Y-%m-%d')} is outside simulation range")
            return pd.DataFrame(columns=['symbol', 'shares', 'price', 'value'])

        # Calculate holdings for the target date by replaying trades up to that date
        # Use shared clipping logic
        holdings_at_date = {symbol: 0 for symbol in self.holdings.keys()}

        for i, trade in enumerate(self.trade_history):
            trade_date = self._cached_trade_dates[i]
            if trade_date <= target_date:
                # Use shared clipping logic method
                actual_volume = self._apply_clipping_logic(trade, i, target_date)
                holdings_at_date[trade.symbol] += actual_volume
            else:
                break

        # Get prices for the target date
        holdings_data = []
        total_value = 0.0

        for symbol, shares in holdings_at_date.items():
            if shares != 0 and symbol in self.symbol_data:
                prices = self.symbol_data[symbol]['Close'].reindex(self.date_range, method='ffill')
                if target_date in prices.index:
                    price = prices.loc[target_date]
                    value = shares * price
                    total_value += value

                    holdings_data.append({
                        'symbol': symbol,
                        'shares': shares,
                        'price': price,
                        'value': value
                    })

        # Get cash for the target date
        cash = self.portfolio_df.loc[target_date, 'cash'] if target_date in self.portfolio_df.index else 0.0
        total_portfolio = total_value + cash

        # Create DataFrame
        holdings_df = pd.DataFrame(holdings_data)

        # Add summary row
        if len(holdings_df) > 0:
            summary_row = pd.DataFrame({
                'symbol': ['CASH', 'TOTAL'],
                'shares': [1, 1],  # Placeholder values
                'price': [cash, total_portfolio],
                'value': [cash, total_portfolio]
            })
            holdings_df = pd.concat([holdings_df, summary_row], ignore_index=True)

        if show_report:
            self._print_holdings_report(target_date, holdings_df, cash, total_value, total_portfolio)

        return holdings_df

    def _print_holdings_report(self, date: pd.Timestamp, holdings_df: pd.DataFrame,
                              cash: float, total_holdings: float, total_portfolio: float):
        """Print a formatted holdings report for a specific date."""
        print(f"\n" + "="*60)
        print(f"HOLDINGS REPORT - {date.strftime('%Y-%m-%d')}")
        print("="*60)
        print(f"{'Symbol':<10} {'Shares':<10} {'Price':<12} {'Value':<15}")
        print("-"*60)

        # Filter out summary rows for the main display (handle empty DataFrame)
        if len(holdings_df) > 0 and 'symbol' in holdings_df.columns:
            position_rows = holdings_df[~holdings_df['symbol'].isin(['CASH', 'TOTAL'])]

            for _, row in position_rows.iterrows():
                print(f"{row['symbol']:<10} {row['shares']:<10.0f} ${row['price']:<11.2f} ${row['value']:<14.2f}")
        else:
            print("No positions held")

        print("-"*60)
        print(f"{'Holdings':<10} {'':<10} {'':<12} ${total_holdings:<14.2f}")
        print(f"{'Cash':<10} {'':<10} {'':<12} ${cash:<14.2f}")
        print("="*60)
        print(f"{'TOTAL':<10} {'':<10} {'':<12} ${total_portfolio:<14.2f}")
        print("="*60)

    def get_portfolio_history(self, symbols: list = None, show_report: bool = False) -> pd.DataFrame:
        """
        Get complete portfolio history across all dates.

        Args:
            symbols (list, optional): Specific symbols to include. If None, includes all.
            show_report (bool): Whether to print a summary report

        Returns:
            pd.DataFrame: Portfolio history with holdings and values over time
        """
        if not hasattr(self, 'portfolio_df') or self.portfolio_df is None:
            return pd.DataFrame()

        # Build complete holdings history
        all_symbols = list(self.holdings.keys()) if symbols is None else symbols
        history_data = []

        for date in self.date_range:
            # Calculate holdings for this date using shared clipping logic
            holdings_at_date = {symbol: 0 for symbol in all_symbols}

            for i, trade in enumerate(self.trade_history):
                trade_date = self._cached_trade_dates[i]
                if trade_date <= date and trade.symbol in all_symbols:
                    # Use shared clipping logic method
                    actual_volume = self._apply_clipping_logic(trade, i, date)
                    holdings_at_date[trade.symbol] += actual_volume

            # Get values for this date
            date_data = {'date': date}
            total_holdings_value = 0.0

            for symbol in all_symbols:
                shares = holdings_at_date[symbol]
                date_data[f'{symbol}_shares'] = shares

                if shares != 0 and symbol in self.symbol_data:
                    prices = self.symbol_data[symbol]['Close'].reindex(self.date_range, method='ffill')
                    if date in prices.index:
                        price = prices.loc[date]
                        value = shares * price
                        date_data[f'{symbol}_price'] = price
                        date_data[f'{symbol}_value'] = value
                        total_holdings_value += value
                    else:
                        date_data[f'{symbol}_price'] = 0.0
                        date_data[f'{symbol}_value'] = 0.0
                else:
                    date_data[f'{symbol}_price'] = 0.0
                    date_data[f'{symbol}_value'] = 0.0

            # Add portfolio totals
            cash = self.portfolio_df.loc[date, 'cash']
            date_data['cash'] = cash
            date_data['total_holdings'] = total_holdings_value
            date_data['total_portfolio'] = total_holdings_value + cash

            history_data.append(date_data)

        history_df = pd.DataFrame(history_data)
        history_df.set_index('date', inplace=True)

        if show_report:
            print(f"\nPortfolio History Summary ({len(history_df)} days)")
            print(f"Symbols tracked: {', '.join(all_symbols)}")
            print(f"Date range: {history_df.index.min().strftime('%Y-%m-%d')} to {history_df.index.max().strftime('%Y-%m-%d')}")
            print(f"Final portfolio value: ${history_df['total_portfolio'].iloc[-1]:,.2f}")

        return history_df