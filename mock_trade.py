from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from backend import Backend
from pandas import DataFrame
import pandas as pd
from structs import Trade, StockConfig


@dataclass
class MockTrade:
    backend: Backend
    trade_history: List[Trade]
    start_date: str
    end_date: str
    name: str = "Portfolio"
    benchmark_symbol: str = "SPY"

    def __post_init__(self):
        self.symbol_data: Dict[str, DataFrame] = {}
        self.portfolio_df: Optional[DataFrame] = None
        self.holdings: Dict[str, int] = {}  # symbol -> shares held
        self.date_range: pd.DatetimeIndex = None
        self.current_trade_index: int = 0
        self.initial_investment: float = 0.0
        self._cached_trade_dates: Dict[int, pd.Timestamp] = {}  # Cache adjusted trade dates
        self._executed_volumes: Dict[int, int] = {}  # Cache actual executed volumes (after clipping)
        self._cached_prices: Dict[str, pd.Series] = {}  # Cache reindexed price data
        self._incremental_holdings: Dict[str, int] = {}  # Track holdings incrementally

    def mock(self, as_stock_config: bool = True) -> Union[DataFrame, StockConfig]:
        """Execute the mock trading simulation and return portfolio performance."""
        self.initialize_mock()

        # Process all trades in order
        while self.current_trade_index < len(self.trade_history):
            self.trade_one()
            self.current_trade_index += 1

        # Calculate portfolio values once at the end for performance
        print(f"\nCalculating portfolio values...")
        self._update_portfolio_values()

        # Show final portfolio state at end of simulation
        end_date = pd.to_datetime(self.end_date)
        if end_date.weekday() >= 5:  # Adjust to business day if needed
            end_date = end_date - pd.Timedelta(days=end_date.weekday() - 4)

        print(f"\n{'='*60}")
        print(f"FINAL PORTFOLIO STATE - {end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        self._update_portfolio_values(end_date)

        # Calculate and display average yearly return
        self._calculate_yearly_return(end_date, self.benchmark_symbol)

        if as_stock_config:
            # Filter out dates with zero portfolio values (non-business days or incomplete data)
            valid_mask = self.portfolio_df['portfolio_value'] > 0
            filtered_portfolio = self.portfolio_df[valid_mask]

            final_portfolio_df = pd.DataFrame({
                'Close': filtered_portfolio['portfolio_value'],
                'Open': filtered_portfolio['portfolio_value'],
                'High': filtered_portfolio['portfolio_value'],
                'Low': filtered_portfolio['portfolio_value']
            }, index=filtered_portfolio.index)
            final_portfolio_df['symbol'] = self.name
            portfolio_config = StockConfig(symbol=self.name, data=final_portfolio_df, normalize=True)
            return portfolio_config
        else:
            return self.portfolio_df

    def initialize_mock(self):
        """
        1. fetch all relevant stock prices across daterange from backend
        2. initialize portfolio data frames
        3. assume the investment amount == sum of stock values in the 1st trade date
        """
        # Get all unique symbols from trade history
        trade_symbols = set(trade.symbol for trade in self.trade_history)

        # Add benchmark symbol to ensure it's available for performance comparison
        trade_symbols.add(self.benchmark_symbol)

        # Fetch price data for all symbols
        for symbol in trade_symbols:
            self.symbol_data[symbol] = self.backend.get_daily_price(symbol, self.start_date, self.end_date)

        # Create business day date range FIRST
        self.date_range = pd.bdate_range(start=self.start_date, end=self.end_date, freq='B')

        # Pre-cache reindexed price data for performance (AFTER date_range is defined)
        self._cache_price_data()

        # Initialize portfolio DataFrame
        self.portfolio_df = pd.DataFrame(index=self.date_range)
        self.portfolio_df.index.name = 'date'
        self.portfolio_df['total_value'] = 0.0
        self.portfolio_df['cash'] = 0.0
        self.portfolio_df['portfolio_value'] = 0.0

        # Initialize holdings
        for symbol in trade_symbols:
            self.holdings[symbol] = 0
            self._incremental_holdings[symbol] = 0

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

                if trade_date == first_trade_date and (trade.volume is not None and trade.volume > 0 or trade.cash_amount is not None and trade.cash_amount > 0 or trade.percentage is not None and trade.percentage > 0):  # Only count buys
                    if trade.symbol in self.symbol_data:
                        prices = self.symbol_data[trade.symbol]['Close'].reindex(self.date_range, method='ffill')
                        if trade_date in prices.index:
                            price = prices.loc[trade_date]
                            if trade.volume is not None:
                                initial_value += trade.volume * price
                            elif trade.cash_amount is not None:
                                initial_value += trade.cash_amount

            self.initial_investment = initial_value
            print(f"Calculated initial investment: ${self.initial_investment:.2f}")

        # Set initial cash
        self.portfolio_df['cash'] = self.initial_investment
        self.portfolio_df['portfolio_value'] = self.initial_investment
        print(f"Portfolio initialized with ${self.initial_investment:.2f} cash")

        # Pre-cache adjusted trade dates for performance
        self._cache_trade_dates()

        # Convert all cash-based trades to volume-based trades upfront
        self._convert_cash_trades()

        # Initialize trade logging table
        self._init_trade_log()

    def _cache_price_data(self):
        """Pre-cache reindexed price data for all symbols to avoid repeated reindexing."""
        for symbol in self.symbol_data:
            # Reindex price data to business days and forward-fill missing values
            price_series = self.symbol_data[symbol]['Close']
            reindexed_prices = price_series.reindex(self.date_range, method='ffill')

            # Ensure no NaN values by using the first available price for early dates
            if reindexed_prices.isna().any():
                first_valid_price = price_series.dropna().iloc[0] if not price_series.dropna().empty else 0
                reindexed_prices = reindexed_prices.fillna(first_valid_price)

            self._cached_prices[symbol] = reindexed_prices

    def _cache_trade_dates(self):
        """Pre-cache adjusted trade dates for performance optimization."""
        for i, trade in enumerate(self.trade_history):
            trade_date = pd.to_datetime(trade.date)
            if trade_date.weekday() >= 5:  # Weekend adjustment
                trade_date = trade_date + pd.Timedelta(days=7 - trade_date.weekday())
            self._cached_trade_dates[i] = trade_date

    def _convert_cash_trades(self):
        """Convert all cash-based trades to volume-based trades using historical prices."""
        for i, trade in enumerate(self.trade_history):
            if trade.cash_amount is not None and trade.volume is None:
                trade_date = self._cached_trade_dates[i]

                # Skip if trade date is outside our range
                if trade_date not in self.date_range:
                    continue

                symbol = trade.symbol
                if symbol in self._cached_prices:
                    # Get price data for this symbol using cached data
                    prices = self._cached_prices[symbol]

                    if trade_date in prices.index:
                        trade_price = prices.loc[trade_date]
                        trade.convert_cash_to_volume(trade_price)

    def _get_symbol_value_at_date(self, symbol: str, target_date: pd.Timestamp) -> float:
        """Get the current position value for a specific symbol at a specific date (before any trades on that date)."""
        # Calculate holdings for this symbol up to the target date
        symbol_holdings = 0

        for i, trade in enumerate(self.trade_history):
            trade_date = self._cached_trade_dates[i]

            # Only process trades for this symbol up to (but not including) the target date
            if trade_date >= target_date:
                break

            if trade.symbol == symbol and trade_date in self.date_range:
                # Use actual volume (which should be converted by now)
                volume = trade.volume or 0
                symbol_holdings += volume

        # Calculate value of current holdings
        if symbol_holdings == 0:
            return 0.0

        if symbol in self.symbol_data:
            # Find the last business day before target_date for pricing
            price_date = target_date - pd.Timedelta(days=1)
            while price_date.weekday() >= 5:  # Skip weekends
                price_date = price_date - pd.Timedelta(days=1)

            # If price date is before our range, use target date
            if price_date < self.date_range[0]:
                price_date = target_date

            prices = self._cached_prices[symbol]
            if price_date in prices.index:
                price = prices.loc[price_date]
                return symbol_holdings * price

        return 0.0

    def _get_portfolio_value_at_date(self, target_date: pd.Timestamp) -> float:
        """Get the portfolio value at a specific date (before any trades on that date)."""
        # For the first trade, use initial investment
        if target_date <= self.date_range[0]:
            return self.initial_investment

        # Find the last business day before target_date
        previous_date = target_date - pd.Timedelta(days=1)
        while previous_date.weekday() >= 5:  # Skip weekends
            previous_date = previous_date - pd.Timedelta(days=1)

        # If previous date is before our range, use initial investment
        if previous_date < self.date_range[0]:
            return self.initial_investment

        # Calculate portfolio value at previous date by simulating trades up to that point
        temp_holdings = {symbol: 0 for symbol in self.holdings.keys()}
        temp_cash = self.initial_investment

        for i, trade in enumerate(self.trade_history):
            trade_date = self._cached_trade_dates[i]

            # Only process trades up to the previous date
            if trade_date > previous_date:
                break

            if trade_date in self.date_range and trade.symbol in self._cached_prices:
                # Get price for this trade using cached data
                prices = self._cached_prices[trade.symbol]
                if trade_date in prices.index:
                    trade_price = prices.loc[trade_date]

                    # Use actual volume (which should be converted by now)
                    volume = trade.volume or 0
                    trade_value = volume * trade_price

                    # Update cash and holdings
                    temp_cash -= trade_value
                    temp_holdings[trade.symbol] += volume

        # Calculate total value at previous date
        total_holdings_value = 0.0
        for symbol, shares in temp_holdings.items():
            if shares != 0 and symbol in self._cached_prices:
                prices = self._cached_prices[symbol]
                if previous_date in prices.index:
                    price = prices.loc[previous_date]
                    total_holdings_value += shares * price

        return total_holdings_value + temp_cash

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
        OPTIMIZED: Uses incremental tracking to avoid recursive calls.

        Args:
            symbol: Symbol to calculate holdings for
            target_date: Date to calculate holdings at
            up_to_trade_index: Only process trades up to this index (exclusive)

        Returns:
            int: Holdings amount at the target date
        """
        holdings = 0
        max_index = up_to_trade_index if up_to_trade_index is not None else len(self.trade_history)

        # Incremental tracking - process trades in order for this symbol only
        for i in range(max_index):
            trade = self.trade_history[i]
            trade_date = self._cached_trade_dates[i]

            # Only count trades for same symbol that happened on or before target date
            if trade.symbol == symbol and trade_date <= target_date:
                actual_volume = trade.volume

                # Apply clipping for sell orders using current holdings
                if trade.volume < 0:  # Sell order
                    max_sellable = -holdings  # Negative because volume is negative for sells
                    if trade.volume < max_sellable:  # Trying to sell more than we have
                        actual_volume = max_sellable

                holdings += actual_volume

        return holdings

    def _apply_clipping_logic(self, trade: Trade, trade_index: int, target_date: pd.Timestamp, current_cash: float = None) -> tuple:
        """
        Apply clipping logic for both sell orders (holdings) and buy orders (cash).

        Args:
            trade: The trade to potentially clip
            trade_index: Index of the current trade (for replay calculation)
            target_date: Date up to which to calculate holdings
            current_cash: Available cash for buy orders (None if not checking cash)

        Returns:
            tuple: (actual_volume, was_clipped, clip_reason)
        """
        actual_volume = trade.volume
        was_clipped = False
        clip_reason = None

        if trade.volume < 0:  # Sell order - check holdings
            # Calculate holdings at the trade date by replaying all previous trades
            holdings_at_date = self._calculate_holdings_at_date(trade.symbol, target_date, trade_index)
            max_sellable = -holdings_at_date  # Negative because volume is negative for sells

            if trade.volume < max_sellable:  # Trying to sell more than we have
                actual_volume = max_sellable
                was_clipped = True
                clip_reason = f"sell_holdings_{holdings_at_date}"

        elif trade.volume > 0 and current_cash is not None:  # Buy order - check cash
            # Get current price for this trade using cached data
            if trade.symbol in self._cached_prices:
                prices = self._cached_prices[trade.symbol]
                if target_date in prices.index:
                    trade_price = prices.loc[target_date]
                    trade_value = trade.volume * trade_price
                    cash_change = -trade_value

                    # Handle infinite values that can't be rounded
                    cash_after_trade = current_cash + cash_change
                    if not (abs(cash_after_trade) < float('inf')) or round(cash_after_trade) < 0:
                        # Ensure we have valid numbers before calculating affordable volume
                        if pd.isna(current_cash) or pd.isna(trade_price) or trade_price <= 0:
                            max_affordable_volume = 0
                        else:
                            max_affordable_volume = int(current_cash / trade_price)

                        if max_affordable_volume < trade.volume:
                            actual_volume = max_affordable_volume
                            was_clipped = True
                            clip_reason = f"buy_cash_{current_cash:.2f}"

        return actual_volume, was_clipped, clip_reason

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

        if symbol in self.symbol_data:
            # Use cached price data for performance
            prices = self._cached_prices[symbol]

            if trade_date in prices.index:
                trade_price = prices.loc[trade_date]

                # Convert percentage-based trades to volume-based using current price and symbol position
                if trade.percentage is not None and trade.volume is None:
                    symbol_value = self._get_symbol_value_at_date(symbol, trade_date)
                    trade.convert_percentage_to_volume(trade_price, symbol_value)

                volume = trade.volume
                action = "BUY" if volume > 0 else "SELL"
                trade_value = volume * trade_price

                # Update cash (subtract for buys, add for sells)
                cash_change = -trade_value
                current_cash = self.portfolio_df.loc[trade_date, 'cash']

                # Debug NaN values
                if pd.isna(current_cash):
                    print(f"DEBUG: current_cash is NaN for {symbol} on {trade_date}")
                    print(f"DEBUG: trade_price={trade_price}, trade_value={trade_value}, cash_change={cash_change}")
                    print(f"DEBUG: portfolio_df cash column stats:")
                    print(self.portfolio_df['cash'].describe())
                if pd.isna(cash_change):
                    print(f"DEBUG: cash_change is NaN for {symbol} on {trade_date}")
                    print(f"DEBUG: trade_price={trade_price}, volume={volume}, trade_value={trade_value}")

                # Apply unified clipping logic for both sell and buy orders
                actual_volume, was_clipped, clip_reason = self._apply_clipping_logic(
                    trade, self.current_trade_index, trade_date, current_cash
                )

                # Update trade record to reflect the actual executed volume
                if was_clipped:
                    trade.volume = actual_volume

                # Update cash change and trade value to match actual volume
                actual_trade_value = actual_volume * trade_price
                cash_change = -actual_trade_value
                trade_value = actual_trade_value

                # Print clipping warnings
                if was_clipped:
                    if clip_reason.startswith("buy_cash"):
                        print(f"WARNING: Buy order clipped - trying to buy {volume} {symbol} but only have ${current_cash:.2f} cash")
                        print(f"         Adjusted to buy {actual_volume} shares instead (${trade_value:.2f})")
                    elif clip_reason.startswith("sell_holdings"):
                        holdings_before_trade = int(clip_reason.split("_")[-1])
                        print(f"WARNING: Sell order clipped - trying to sell {abs(volume)} {symbol} but only have {holdings_before_trade} shares")
                        print(f"         Adjusted to sell {abs(actual_volume)} shares instead")

                # Store the actual executed volume for portfolio calculations
                self._executed_volumes[self.current_trade_index] = actual_volume

                # Update holdings
                self.holdings[symbol] += actual_volume
                self._incremental_holdings[symbol] += actual_volume

                # Efficiently update cash using vectorized operations
                mask = self.portfolio_df.index >= trade_date
                self.portfolio_df.loc[mask, 'cash'] += cash_change

                # Format table row logging
                new_cash = current_cash + cash_change
                holdings_str = f"{symbol}:{self.holdings[symbol]}"
                clipped_note = " (CLIPPED)" if actual_volume != volume else ""
                print(f"{self.current_trade_index+1:<3} {trade_date.strftime('%Y-%m-%d'):<12} {action:<6} {symbol:<8} {abs(actual_volume):<8} ${trade_price:<9.2f} ${abs(trade_value):<11.2f} ${new_cash:<11.2f} {holdings_str:<15}{clipped_note}")

        # Note: Portfolio value updates and holdings summary are deferred to the end for performance

    def _update_portfolio_values(self, current_trade_date: pd.Timestamp = None):
        """Update total portfolio values based on holdings at each date - optimized version."""
        # Use pre-cached price data for all symbols (major optimization)
        cached_prices = self._cached_prices

        # Build holdings using vectorized operations
        holdings_df = pd.DataFrame(0, index=self.date_range, columns=list(self.holdings.keys()))

        # Process trades efficiently using cached dates and executed volumes
        max_trade_index = min(self.current_trade_index + 1, len(self.trade_history))
        for i in range(max_trade_index):
            trade = self.trade_history[i]
            trade_date = self._cached_trade_dates[i]  # Use cached date

            if trade_date not in self.date_range:
                continue

            # Use actual executed volume (after clipping) instead of original trade volume
            if i in self._executed_volumes:
                executed_volume = self._executed_volumes[i]
            else:
                executed_volume, _, _ = self._apply_clipping_logic(trade, i, trade_date)
                # For consistency, also cache this calculated volume
                self._executed_volumes[i] = executed_volume

            # Vectorized update - much faster than loop
            mask = holdings_df.index >= trade_date
            holdings_df.loc[mask, trade.symbol] += executed_volume

        # Vectorized portfolio value calculation
        total_values = pd.Series(0.0, index=self.date_range)

        for symbol in holdings_df.columns:
            if symbol in cached_prices:
                # Vectorized multiplication - much faster
                position_values = holdings_df[symbol] * cached_prices[symbol]
                # Ensure no NaN values are introduced
                position_values = position_values.fillna(0)
                # Only add valid numeric values
                total_values += position_values

        # Batch update portfolio DataFrame
        self.portfolio_df['total_value'] = total_values
        self.portfolio_df['portfolio_value'] = self.portfolio_df['total_value'] + self.portfolio_df['cash']

        # Log current holdings after portfolio update
        print("\nCurrent Holdings Summary:")
        print("-" * 40)
        total_holding_value = 0.0
        for symbol, shares in self.holdings.items():
            if shares != 0 and symbol in cached_prices:
                # Use price at current trade date if available, otherwise latest price
                if current_trade_date is not None and current_trade_date in cached_prices[symbol].index:
                    current_price = cached_prices[symbol].loc[current_trade_date]
                else:
                    current_price = cached_prices[symbol].iloc[-1]  # Fallback to latest price
                position_value = shares * current_price
                total_holding_value += position_value
                print(f"{symbol:<8}: {shares:>6} shares @ ${current_price:>8.2f} = ${position_value:>10.2f}")

        # Use cash at current trade date if available, otherwise latest cash
        if current_trade_date is not None and current_trade_date in self.portfolio_df.index:
            current_cash = self.portfolio_df.loc[current_trade_date, 'cash']
        else:
            current_cash = self.portfolio_df['cash'].iloc[-1]
        total_portfolio = total_holding_value + current_cash

        print("-" * 40)
        print(f"{'Total':<8}: ${total_holding_value:>26.2f}")
        print(f"{'Cash':<8}: ${current_cash:>26.2f}")
        print(f"{'Portfolio':<8}: ${total_portfolio:>26.2f}")
        print("-" * 40)

    def _calculate_yearly_return(self, end_date: pd.Timestamp, benchmark_symbol: str = "SPY"):
        """Calculate and display the average yearly return of the portfolio."""
        start_date = pd.to_datetime(self.start_date)

        # Calculate time period in years
        time_diff = end_date - start_date
        years = time_diff.days / 365.25  # Account for leap years

        # Get initial and final portfolio values
        initial_value = self.initial_investment

        # Get final portfolio value at end date
        if end_date in self.portfolio_df.index:
            final_value = self.portfolio_df.loc[end_date, 'portfolio_value']
        else:
            # Find closest date if end date not in index
            final_value = self.portfolio_df['portfolio_value'].iloc[-1]

        # Calculate total return and annualized return
        total_return = (final_value - initial_value) / initial_value

        if years > 0:
            # Compound annual growth rate (CAGR)
            annualized_return = (final_value / initial_value) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        print(f"\n{'='*40}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*40}")
        print(f"Investment Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Time Period: {years:.2f} years")
        print(f"Initial Investment: ${initial_value:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%} (${final_value - initial_value:,.2f})")
        print(f"Average Yearly Return (CAGR): {annualized_return:.2%}")

        # Calculate benchmark performance
        try:
            benchmark_data = self.backend.get_daily_price(benchmark_symbol, self.start_date, self.end_date)
            benchmark_start_price = benchmark_data['Close'].iloc[0]
            benchmark_end_price = benchmark_data['Close'].iloc[-1]
            benchmark_total_return = (benchmark_end_price - benchmark_start_price) / benchmark_start_price

            if years > 0:
                benchmark_annualized_return = (benchmark_end_price / benchmark_start_price) ** (1 / years) - 1
            else:
                benchmark_annualized_return = 0.0

            print(f"\nBENCHMARK ({benchmark_symbol}):")
            print(f"{benchmark_symbol} Total Return: {benchmark_total_return:.2%}")
            print(f"{benchmark_symbol} Average Yearly Return (CAGR): {benchmark_annualized_return:.2%}")

            # Calculate relative performance
            excess_return = annualized_return - benchmark_annualized_return
            print(f"Excess Return vs {benchmark_symbol}: {excess_return:.2%}")

        except Exception as e:
            print(f"\nNote: Could not fetch {benchmark_symbol} benchmark data: {e}")

        print(f"{'='*40}")

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
                # Use actual executed volume if available, otherwise calculate clipping
                if i in self._executed_volumes:
                    actual_volume = self._executed_volumes[i]
                else:
                    actual_volume, _, _ = self._apply_clipping_logic(trade, i, target_date)
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
                    # Use actual executed volume if available, otherwise calculate clipping
                    if i in self._executed_volumes:
                        actual_volume = self._executed_volumes[i]
                    else:
                        actual_volume, _, _ = self._apply_clipping_logic(trade, i, date)
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