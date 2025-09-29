import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os
from database import Database


class Backend:
    """Backend class for stock data access using existing Database only."""

    def __init__(self, database: Database):
        """
        Initialize the Backend with Database-only access.

        Args:
            database (Database): A database to visit data from
        """
        self.database = database

        print(f"✓ Backend initialized with database-only access: {database.file_path}")

    def _get_database(self) -> Database:
        return self.database

    def _get_data_from_database(self, symbol: str, start_date: str, end_date: str, source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get data from database with proper error handling."""
        try:
            database = self._get_database()
            return database.get_cached_data(symbol, start_date, end_date, source=source)
        except Exception as e:
            raise Exception(f"Failed to access database: {e}")

    def has_data_for_period(self, symbol: str, start_date: str, end_date: str, source: Optional[str] = None) -> bool:
        """Check if data exists for the specified period."""
        try:
            database = self._get_database()
            return database.has_data(symbol, start_date, end_date, source=source)
        except Exception:
            return False

    def get_daily_price(self, symbol: str, start_date: str, end_date: str, source: Optional[str] = None, normalize: bool = False) -> pd.DataFrame:
        """
        Get daily price data from database only.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            source (str, optional): Data source to filter by. If None, searches all sources.
            normalize (bool): If True, normalize to percentage change from first value. Default is False.

        Returns:
            pd.DataFrame: Daily price data, optionally normalized

        Raises:
            ValueError: If inputs are invalid or data not found
        """
        # Validate inputs
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if not start_date:
            raise ValueError("Start date is required")
        if not end_date:
            raise ValueError("End date is required")

        # Get data from database
        data = self._get_data_from_database(symbol, start_date, end_date, source=source)

        if data is None or len(data) == 0:
            available_symbols = self.get_available_symbols()
            if symbol.upper() not in [s.upper() for s in available_symbols]:
                raise ValueError(
                    f"No data found for symbol {symbol.upper()}. "
                    f"Available symbols: {', '.join(available_symbols[:10])}{'...' if len(available_symbols) > 10 else ''}"
                )
            else:
                # Symbol exists but no data for date range
                date_range = self.get_date_range_for_symbol(symbol, source=source)
                raise ValueError(
                    f"No data found for {symbol.upper()} in range {start_date} to {end_date}. "
                    f"Available data range: {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}"
                )

        print(f"📂 Retrieved {len(data)} days of data for {symbol.upper()} from database")
        formatted_data = self._format_output_dataframe(data)

        # Apply normalization if requested
        if normalize:
            formatted_data = self._normalize_data(formatted_data, 'first')

        return formatted_data

    def get_date_range_for_symbol(self, symbol: str, source: Optional[str] = None) -> Dict[str, str]:
        """Get available date range for a specific symbol."""
        try:
            database = self._get_database()
            return database.get_date_range(symbol=symbol, source=source)
        except Exception:
            return {'start_date': None, 'end_date': None}

    def _format_output_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame for output with consistent column selection and naming.

        Args:
            df (pd.DataFrame): Raw DataFrame with price data

        Returns:
            pd.DataFrame: Formatted DataFrame with selected columns
        """
        # Define column mapping for backward compatibility
        column_mapping = {
            'Adj_Open': 'Adj Open',
            'Adj_High': 'Adj High',
            'Adj_Low': 'Adj Low',
            'Adj_Close': 'Adj Close',
            'Adj_Volume': 'Adj Volume',
            'Split_Factor': 'Split Factor'
        }

        # Apply column mapping if columns exist
        df_formatted = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_formatted.columns:
                df_formatted = df_formatted.rename(columns={old_col: new_col})

        # Select main OHLCV columns in standard order
        main_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Split Factor']
        available_columns = [col for col in main_columns if col in df_formatted.columns]

        # Add adjusted columns if available (with backward-compatible names)
        adj_columns = ['Adj Close', 'Adj Volume']
        available_adj_columns = [col for col in adj_columns if col in df_formatted.columns]

        final_columns = available_columns + available_adj_columns
        return df_formatted[final_columns].copy() if final_columns else df_formatted

    def _normalize_data(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Normalize price data to percentage change from first value.

        Args:
            df (pd.DataFrame): DataFrame with price data
            method (str): Normalization method (only 'first' is supported)

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        # Only normalize numeric price columns, preserve volume and other columns as-is
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        numeric_columns = [col for col in price_columns if col in df.columns]

        if not numeric_columns:
            return df

        df_normalized = df.copy()

        # Normalize to percentage change from first value
        for col in numeric_columns:
            first_val = df[col].iloc[0]
            if first_val != 0:  # Avoid division by zero
                df_normalized[col] = df[col] / first_val
            else:
                df_normalized[col] = 1

        return df_normalized

    def get_available_symbols(self) -> List[str]:
        """Get all symbols available in the database."""
        try:
            database = self._get_database()
            return database.get_symbols()
        except Exception:
            return []

    def get_data_summary(self, symbol: str = None, source: Optional[str] = None) -> Dict:
        """Get summary statistics for stored data."""
        try:
            database = self._get_database()
            return database.get_summary_stats(symbol=symbol, source=source)
        except Exception:
            return {}

    def export_data(self, output_path: str, symbol: str = None,
                   start_date: str = None, end_date: str = None, source: Optional[str] = None) -> str:
        """Export data to CSV file."""
        try:
            database = self._get_database()
            return database.export_to_csv(output_path, symbol=symbol, source=source,
                                        start_date=start_date, end_date=end_date)
        except Exception as e:
            raise Exception(f"Export failed: {e}")

    def database_info(self) -> None:
        """Print database information."""
        try:
            database = self._get_database()
            database.info()
        except Exception as e:
            print(f"📊 Database access error: {e}")

    def check_data_coverage(self, symbol: str = None, source: Optional[str] = None) -> Dict:
        """
        Check data coverage for symbols in the database.

        Args:
            symbol (str, optional): Check specific symbol, or all if None
            source (str, optional): Data source to filter by. If None, searches all sources.

        Returns:
            Dict: Coverage information
        """
        try:
            database = self._get_database()

            if symbol:
                # Check specific symbol
                symbols_to_check = [symbol.upper()]
            else:
                # Check all symbols
                symbols_to_check = database.get_symbols()

            coverage = {}
            for sym in symbols_to_check:
                date_range = database.get_date_range(symbol=sym, source=source)
                symbol_data = database.query(symbol=sym, source=source)

                coverage[sym] = {
                    'start_date': date_range.get('start_date'),
                    'end_date': date_range.get('end_date'),
                    'total_days': len(symbol_data) if symbol_data is not None else 0,
                    'has_data': date_range.get('start_date') is not None
                }

            return coverage

        except Exception as e:
            print(f"Error checking data coverage: {e}")
            return {}

    def get_latest_data(self, symbol: str, n_days: int = 5, source: Optional[str] = None) -> pd.DataFrame:
        """
        Get the most recent n days of data for a symbol.

        Args:
            symbol (str): Stock symbol
            n_days (int): Number of recent days to retrieve
            source (str, optional): Data source to filter by. If None, searches all sources.

        Returns:
            pd.DataFrame: Latest data

        Raises:
            ValueError: If symbol has no data
        """
        try:
            database = self._get_database()
            data = database.get_latest_data(symbol=symbol, source=source, n_days=n_days)

            if data is None or len(data) == 0:
                raise ValueError(f"No recent data found for {symbol.upper()}")

            # Remove metadata columns and format output
            return self._format_output_dataframe(data)

        except Exception as e:
            raise Exception(f"Failed to get latest data for {symbol.upper()}: {e}")
