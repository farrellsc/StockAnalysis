from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

from structs import Trade


class BaseStrategy:
    """Base class for all trading strategies."""

    def get_trades(self) -> List[Trade]:
        """
        Generate a list of trades based on the strategy.

        Returns:
            List[Trade]: List of trades to execute
        """
        raise NotImplementedError("Subclasses must implement get_trades method")


@dataclass
class DollarCostAverageStrategy(BaseStrategy):
    """
    Dollar Cost Averaging strategy that invests either a fixed amount or percentage at regular intervals.

    This strategy helps reduce the impact of volatility by spreading purchases over time,
    potentially lowering the average cost per share compared to making a single large purchase.

    Two investment modes are supported:
    1. Fixed amount mode: Specify cash_amount and end_date
    2. Percentage mode: Specify percentage (uses Trade.percentage for portfolio-based investments)

    Attributes:
        symbol (str): Stock symbol to invest in (e.g., 'AAPL', 'SPY')
        frequency (int): Frequency of investments in days (e.g., 30 for monthly, 7 for weekly)
        start_date (str): Start date for investments in 'YYYY-MM-DD' format
        cash_amount (Optional[float]): Amount to invest per period in dollars (fixed amount mode)
        end_date (Optional[str]): End date for investments in 'YYYY-MM-DD' format (fixed amount mode)
        percentage (Optional[float]): Percentage of portfolio to invest per period (percentage mode)

    Examples:
        Fixed amount mode (invest $1000 monthly until end date):
        >>> strategy = DollarCostAverageStrategy(
        ...     symbol='SPY',
        ...     cash_amount=1000.0,
        ...     frequency=30,  # Monthly
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31'
        ... )
        >>> trades = strategy.get_trades()

        Percentage mode (invest 10% of portfolio monthly):
        >>> strategy = DollarCostAverageStrategy(
        ...     symbol='SPY',
        ...     percentage=10.0,  # 10% of portfolio
        ...     frequency=30,  # Monthly
        ...     start_date='2023-01-01'
        ... )
        >>> trades = strategy.get_trades()
    """
    symbol: str
    frequency: int
    start_date: str
    cash_amount: Optional[float] = None
    end_date: Optional[str] = None
    percentage: Optional[float] = None

    def __post_init__(self):
        """Validate investment parameters."""
        # Check that exactly one investment mode is specified
        fixed_mode = self.cash_amount is not None and self.end_date is not None
        percent_mode = self.percentage is not None

        if fixed_mode and percent_mode:
            raise ValueError("Cannot specify both fixed amount mode (cash_amount + end_date) and percentage mode (percentage)")

        if not fixed_mode and not percent_mode:
            raise ValueError("Must specify either fixed amount mode (cash_amount + end_date) or percentage mode (percentage)")

        # Validate percentage mode parameters
        if percent_mode:
            if not (0 < self.percentage <= 100):
                raise ValueError(f"percentage must be between 0 and 100, got {self.percentage}")
            if self.end_date is not None:
                raise ValueError("end_date should not be specified in percentage mode")

        # Validate fixed amount mode parameters
        if fixed_mode:
            if self.cash_amount <= 0:
                raise ValueError(f"cash_amount must be positive, got {self.cash_amount}")
            if self.percentage is not None:
                raise ValueError("percentage should not be specified in fixed amount mode")

    def get_trades(self) -> List[Trade]:
        """
        Generate dollar cost averaging trades based on the investment mode.

        Fixed Amount Mode:
        Creates trades at regular intervals from start_date to end_date,
        investing the same cash amount each time.

        Percentage Mode:
        Creates a single trade using Trade.percentage parameter, which will be processed
        by the trading system to calculate actual amounts based on portfolio value.

        Only creates trades on business days (Monday-Friday), automatically adjusting
        weekend dates to the next business day.

        Returns:
            List[Trade]: List of trades with either cash_amount (fixed mode) or percentage (percentage mode)

        Raises:
            ValueError: If start_date format is invalid, or if start_date is after end_date (fixed mode)

        Examples:
            Fixed amount mode - monthly ($1000) investments in SPY from Jan 2023 to Mar 2023:
            - Trade 1: 2023-01-01, $1000 in SPY
            - Trade 2: 2023-01-31, $1000 in SPY
            - Trade 3: 2023-03-02, $1000 in SPY (adjusted from weekend)

            Percentage mode - 10% of portfolio monthly:
            - Trade 1: 2023-01-01, 10% of portfolio in SPY
            - Trade 2: 2023-01-31, 10% of portfolio in SPY
            - Trade 3: 2023-03-02, 10% of portfolio in SPY
        """
        # Determine investment mode
        if self.cash_amount is not None:
            return self._get_trades_fixed_amount()
        else:
            return self._get_trades_percentage()

    def _get_trades_fixed_amount(self) -> List[Trade]:
        """Generate trades for fixed amount mode."""
        trades = []

        # Parse and validate dates
        try:
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD'. Error: {e}")

        if start_dt > end_dt:
            raise ValueError(f"Start date {self.start_date} must be before end date {self.end_date}")

        # Generate trade dates at regular frequency intervals
        current_date = start_dt
        trade_index = 0

        while current_date <= end_dt:
            # Adjust to next business day if current date is weekend
            adjusted_date = self._adjust_to_business_day(current_date)

            # Only create trade if adjusted date is still within our end date range
            if adjusted_date <= end_dt:
                trade = Trade(
                    index=trade_index,
                    symbol=self.symbol,
                    date=adjusted_date.strftime('%Y-%m-%d'),
                    cash_amount=float(self.cash_amount),
                    desc=f"DCA #{trade_index + 1}: ${self.cash_amount} in {self.symbol}"
                )
                trades.append(trade)
                trade_index += 1

            # Move to next investment date
            current_date += timedelta(days=self.frequency)

        return trades

    def _get_trades_percentage(self) -> List[Trade]:
        """Generate trades for percentage mode using Trade.percentage parameter until 100% is invested."""
        trades = []

        # Parse start date
        try:
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD'. Error: {e}")

        # Generate trades until cumulative percentage reaches 100%
        current_date = start_dt
        trade_index = 0
        cumulative_percentage = 0.0
        max_trades = 10000  # Safety limit to prevent infinite generation

        while cumulative_percentage < 100.0 and trade_index < max_trades:
            # Calculate remaining percentage that can be invested
            remaining_percentage = 100.0 - cumulative_percentage

            # Use the smaller of our target percentage or remaining percentage
            actual_percentage = min(self.percentage, remaining_percentage)

            # Stop if the remaining percentage is too small to be meaningful
            if actual_percentage < 0.01:  # Less than 0.01%
                break

            # Adjust to next business day if current date is weekend
            adjusted_date = self._adjust_to_business_day(current_date)

            # Create trade using percentage parameter
            trade = Trade(
                index=trade_index,
                symbol=self.symbol,
                date=adjusted_date.strftime('%Y-%m-%d'),
                percentage=actual_percentage,
                desc=f"DCA #{trade_index + 1}: {actual_percentage}% of portfolio in {self.symbol} (cumulative: {cumulative_percentage + actual_percentage:.1f}%)"
            )
            trades.append(trade)

            # Update cumulative percentage and counters
            cumulative_percentage += actual_percentage
            trade_index += 1

            # Move to next investment date
            current_date += timedelta(days=self.frequency)

        return trades

    def _adjust_to_business_day(self, date: datetime) -> datetime:
        """
        Adjust a date to the next business day if it falls on a weekend.

        Args:
            date (datetime): Date to potentially adjust

        Returns:
            datetime: Date adjusted to next Monday if weekend, otherwise unchanged

        Note:
            Monday=0, Tuesday=1, ..., Saturday=5, Sunday=6
        """
        # If it's Saturday (5) or Sunday (6), move to next Monday
        if date.weekday() >= 5:
            days_to_add = 7 - date.weekday()  # Saturday: 2 days, Sunday: 1 day
            return date + timedelta(days=days_to_add)
        return date