import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


class Database:
    """
    A database interface for accessing pickled DataFrame stock data files.

    Provides a comprehensive API for querying, filtering, and analyzing
    stock data stored in the format created by the Crawler class.
    """

    def __init__(self, file_path: str):
        """
        Initialize the Database with a pickled DataFrame file.

        Args:
            file_path (str): Path to the pickled DataFrame file

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If the file cannot be loaded or has wrong format
        """
        self.file_path = Path(file_path)
        self._df = None
        self._last_modified = None

        if not self.file_path.exists():
            raise FileNotFoundError(f"Database file not found: {file_path}")

        # Load the data and validate format
        self._load_data()
        self._validate_format()

        print(f"✓ Database initialized: {len(self._df)} rows from {self.file_path}")

    def _load_data(self) -> None:
        """Load data from the pickle file with caching based on file modification time."""
        current_modified = self.file_path.stat().st_mtime

        # Only reload if file has been modified
        if self._last_modified != current_modified:
            try:
                with open(self.file_path, 'rb') as f:
                    self._df = pickle.load(f)
                self._last_modified = current_modified
                print(f"📂 Data loaded: {len(self._df)} rows")
            except Exception as e:
                raise Exception(f"Failed to load database file: {e}")

    def _validate_format(self) -> None:
        """Validate that the DataFrame has the expected format from crawler."""
        if not isinstance(self._df, pd.DataFrame):
            raise Exception("File does not contain a pandas DataFrame")

        required_columns = {'symbol', 'source', 'last_updated'}
        missing_columns = required_columns - set(self._df.columns)

        if missing_columns:
            raise Exception(f"DataFrame missing required columns: {missing_columns}")

        if not isinstance(self._df.index, pd.DatetimeIndex):
            raise Exception("DataFrame must have a DatetimeIndex")

    def refresh(self) -> None:
        """Force reload data from file if it has been modified."""
        self._load_data()

    def get_symbols(self) -> List[str]:
        """
        Get all unique symbols in the database.

        Returns:
            List[str]: Sorted list of unique symbols
        """
        return sorted(self._df['symbol'].unique().tolist())

    def get_sources(self) -> List[str]:
        """
        Get all unique data sources in the database.

        Returns:
            List[str]: Sorted list of unique sources
        """
        return sorted(self._df['source'].unique().tolist())

    def get_date_range(self, symbol: Optional[str] = None, source: Optional[str] = None) -> Dict[str, str]:
        """
        Get the date range of available data.

        Args:
            symbol (str, optional): Filter by symbol
            source (str, optional): Filter by source

        Returns:
            Dict[str, str]: Dictionary with 'start_date' and 'end_date'
        """
        df = self._df

        if symbol:
            df = df[df['symbol'] == symbol.upper()]
        if source:
            df = df[df['source'] == source.lower()]

        if len(df) == 0:
            return {'start_date': None, 'end_date': None}

        return {
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d')
        }

    def query(self, symbol: Optional[Union[str, List[str]]] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              source: Optional[str] = None,
              columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query data with flexible filtering options.

        Args:
            symbol (str or List[str], optional): Symbol(s) to filter by
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            source (str, optional): Data source to filter by
            columns (List[str], optional): Specific columns to return

        Returns:
            pd.DataFrame: Filtered data
        """
        df = self._df.copy()

        # Apply filters
        if symbol:
            if isinstance(symbol, str):
                df = df[df['symbol'] == symbol.upper()]
            else:  # List of symbols
                df = df[df['symbol'].isin([s.upper() for s in symbol])]

        if source:
            df = df[df['source'] == source.lower()]

        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Handle timezone-aware index
            if df.index.tz is not None:
                start_dt = start_dt.tz_localize(df.index.tz)
            df = df[df.index >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Handle timezone-aware index
            if df.index.tz is not None:
                end_dt = end_dt.tz_localize(df.index.tz)
            df = df[df.index <= end_dt]

        # Select specific columns if requested
        if columns:
            available_columns = [col for col in columns if col in df.columns]
            if available_columns:
                df = df[available_columns]

        return df

    def get_latest_data(self, symbol: Optional[str] = None,
                       source: Optional[str] = None,
                       n_days: int = 1) -> pd.DataFrame:
        """
        Get the most recent data points.

        Args:
            symbol (str, optional): Filter by symbol
            source (str, optional): Filter by source
            n_days (int): Number of latest days to retrieve

        Returns:
            pd.DataFrame: Latest data
        """
        df = self.query(symbol=symbol, source=source)
        return df.tail(n_days)

    def get_price_data(self, symbol: str, source: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get OHLCV price data for a specific symbol.

        Args:
            symbol (str): Stock symbol
            source (str, optional): Data source
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Price data with OHLCV columns
        """
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = self.query(symbol=symbol, source=source, start_date=start_date, end_date=end_date)

        # Return only price columns that exist
        available_price_cols = [col for col in price_columns if col in df.columns]
        return df[available_price_cols]

    def get_summary_stats(self, symbol: Optional[str] = None,
                         source: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for the data.

        Args:
            symbol (str, optional): Filter by symbol
            source (str, optional): Filter by source

        Returns:
            Dict[str, Any]: Summary statistics
        """
        df = self.query(symbol=symbol, source=source)

        if len(df) == 0:
            return {'error': 'No data found for the specified filters'}

        numeric_columns = df.select_dtypes(include=['number']).columns

        summary = {
            'total_records': len(df),
            'symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'sources': df['source'].nunique() if 'source' in df.columns else 0,
            'date_range': self.get_date_range(symbol=symbol, source=source),
            'numeric_stats': df[numeric_columns].describe().to_dict() if len(numeric_columns) > 0 else {}
        }

        return summary

    def search_by_date(self, date: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for a specific date.

        Args:
            date (str): Date in 'YYYY-MM-DD' format
            symbol (str, optional): Filter by symbol

        Returns:
            pd.DataFrame: Data for the specified date
        """
        date_filter = self._df.index == pd.to_datetime(date)
        df = self._df[date_filter]

        if symbol:
            df = df[df['symbol'] == symbol.upper()]

        return df

    def info(self) -> None:
        """Print comprehensive information about the database."""
        print(f"\n📊 Database Info: {self.file_path}")
        print(f"├── Total Records: {len(self._df):,}")
        print(f"├── Symbols: {len(self.get_symbols())} ({', '.join(self.get_symbols()[:5])}{'...' if len(self.get_symbols()) > 5 else ''})")
        print(f"├── Sources: {len(self.get_sources())} ({', '.join(self.get_sources())})")

        date_range = self.get_date_range()
        print(f"├── Date Range: {date_range['start_date']} to {date_range['end_date']}")

        print(f"├── Columns: {len(self._df.columns)}")
        for col in self._df.columns:
            print(f"│   ├── {col}")

        print(f"└── File Size: {self.file_path.stat().st_size / (1024*1024):.2f} MB")

    def __len__(self) -> int:
        """Return the number of records in the database."""
        return len(self._df)

    def __repr__(self) -> str:
        """String representation of the Database."""
        return f"Database(file='{self.file_path}', records={len(self)})"

    def save_dataframe(self, df: pd.DataFrame) -> None:
        """
        Save a new DataFrame to the database file, replacing existing data.

        Args:
            df (pd.DataFrame): DataFrame to save

        Raises:
            Exception: If save operation fails
        """
        try:
            # Validate the DataFrame format before saving
            self._validate_dataframe(df)

            with open(self.file_path, 'wb') as f:
                pickle.dump(df, f)

            # Update internal state
            self._df = df.copy()
            self._last_modified = self.file_path.stat().st_mtime

            print(f"💾 Database saved: {len(df)} rows to {self.file_path}")

        except Exception as e:
            raise Exception(f"Failed to save database: {e}")

    def update_data(self, new_df: pd.DataFrame) -> None:
        """
        Update database by merging new data with existing data.
        Removes duplicates for same symbol/source/date combinations using vectorized operations.

        Args:
            new_df (pd.DataFrame): New data to merge

        Raises:
            Exception: If update operation fails
        """
        try:
            # Validate new data format
            self._validate_dataframe(new_df)

            # Normalize timezone compatibility before operations
            new_df = self._normalize_timezone(new_df)

            if len(self._df) > 0:
                # Ensure existing data also has normalized timezone
                existing_df = self._normalize_timezone(self._df)

                # Create composite keys for fast duplicate detection
                # Format: "SYMBOL|source|YYYY-MM-DD"
                existing_keys = (existing_df['symbol'] + '|' +
                               existing_df['source'] + '|' +
                               existing_df.index.strftime('%Y-%m-%d'))

                new_keys = (new_df['symbol'] + '|' +
                           new_df['source'] + '|' +
                           new_df.index.strftime('%Y-%m-%d'))

                # Remove duplicates in one vectorized operation
                mask = ~existing_keys.isin(new_keys)
                filtered_existing = existing_df[mask]

                # Combine with new data
                consolidated_df = pd.concat([filtered_existing, new_df])
                print(f"📊 Merged data: {len(filtered_existing)} existing + {len(new_df)} new = {len(consolidated_df)} total rows")
            else:
                consolidated_df = new_df
                print(f"📊 Created new dataset with {len(consolidated_df)} rows")

            # Sort and save
            consolidated_df = consolidated_df.sort_index()
            self.save_dataframe(consolidated_df)

        except Exception as e:
            raise Exception(f"Failed to update database: {e}")

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timezone handling for DataFrame index.
        Ensures all data uses timezone-naive DatetimeIndex for consistent operations.
        This prevents timezone comparison errors during concatenation and sorting.

        Args:
            df (pd.DataFrame): DataFrame to normalize

        Returns:
            pd.DataFrame: DataFrame with timezone-naive DatetimeIndex
        """
        df_copy = df.copy()

        if isinstance(df_copy.index, pd.DatetimeIndex):
            if df_copy.index.tz is not None:
                # Convert timezone-aware index to timezone-naive by removing timezone info
                # This preserves the actual datetime values while removing timezone complexity
                df_copy.index = df_copy.index.tz_localize(None)

        return df_copy

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame format for saving."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        required_columns = {'symbol', 'source', 'last_updated'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def has_data(self, symbol: str, start_date: str, end_date: str, source: str = None) -> bool:
        """
        Check if data exists for the specified parameters.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            source (str, optional): Data source filter

        Returns:
            bool: True if data exists for the range
        """
        try:
            df = self.query(symbol=symbol, source=source, start_date=start_date, end_date=end_date)
            return df is not None and len(df) > 0
        except:
            return False

    def get_price_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get available price-related columns from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to check

        Returns:
            List[str]: List of available price columns
        """
        standard_price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        adjusted_cols = ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']
        other_cols = ['Split_Factor', 'Dividend']

        available_cols = []
        for col_group in [standard_price_cols, adjusted_cols, other_cols]:
            available_cols.extend([col for col in col_group if col in df.columns])

        return available_cols

    def get_split_history(self, symbol: str) -> pd.DataFrame:
        """
        Get split history for a specific symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            pd.DataFrame: DataFrame with dates and split factors where splits occurred
        """
        symbol_data = self.query(symbol=symbol)

        if symbol_data is None or len(symbol_data) == 0:
            return pd.DataFrame(columns=['date', 'split_factor'])

        # Check for different split factor column names
        split_col = 'Split_Factor'

        if split_col is None:
            return pd.DataFrame(columns=['date', 'split_factor'])

        # Find dates where split factor != 1 (includes both > 1 and < 1 splits)
        splits = symbol_data[symbol_data[split_col] != 1.0].copy()

        if len(splits) == 0:
            return pd.DataFrame(columns=['date', 'split_factor'])

        # Create result DataFrame with date as index
        result = pd.DataFrame({
            'split_factor': splits[split_col]
        })
        result.index.name = 'date'

        return result.sort_index()

    def backfill_split_adjusted_prices(self, symbol: str) -> None:
        """
        Backfill historical stock prices for a symbol by adjusting for splits.
        Historical prices before split dates are divided by split factors.

        Args:
            symbol (str): Stock symbol to adjust

        Example:
            If prices are [6,6,3,3,3] with split_factor=2.0 on 3rd date,
            adjusted prices become [3,3,3,3,3]
        """
        symbol_data = self.query(symbol=symbol)

        if symbol_data is None or len(symbol_data) == 0:
            print(f"No data found for symbol {symbol}")
            return

        # Get split history
        splits = self.get_split_history(symbol)

        if len(splits) == 0:
            print(f"No splits found for {symbol}")
            return

        # Work on a copy of the data
        adjusted_data = symbol_data.copy()

        # Price columns to adjust
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close']
        available_price_cols = [col for col in price_columns if col in adjusted_data.columns]

        if not available_price_cols:
            print(f"No price columns found for {symbol}")
            return

        print(f"Adjusting {symbol} for {len(splits)} splits on columns: {available_price_cols}")

        # Process splits from most recent to oldest to maintain cumulative adjustment
        splits_sorted = splits.sort_index(ascending=False)

        for split_date, split_row in splits_sorted.iterrows():
            split_factor = split_row['split_factor']

            # Adjust all prices before the split date
            before_split_mask = adjusted_data.index < split_date

            for col in available_price_cols:
                adjusted_data.loc[before_split_mask, col] = adjusted_data.loc[before_split_mask, col] / split_factor

            print(f"  Applied {split_factor}x split on {split_date.strftime('%Y-%m-%d')}")

        # Clean up Split_Factor field - set all to 1.0 after adjustment
        split_col = 'Split_Factor'
        adjusted_data[split_col] = 1.0
        print(f"  Cleaned up {split_col} field - set all values to 1.0")

        # Update the internal DataFrame
        symbol_mask = self._df['symbol'] == symbol.upper()

        # Remove old data for this symbol
        self._df = self._df[~symbol_mask]

        # Add adjusted data
        self._df = pd.concat([self._df, adjusted_data])
        self._df = self._df.sort_index()

        # Save to file
        self.save_dataframe(self._df)

        print(f"✓ Split-adjusted prices saved for {symbol}")

    def backfill_all_split_adjusted_prices(self) -> None:
        """
        Apply split adjustments to all symbols in the database.
        """
        symbols = self.get_symbols()

        if not symbols:
            print("No symbols found in database")
            return

        print(f"Applying split adjustments to {len(symbols)} symbols...")

        adjusted_count = 0
        for symbol in symbols:
            try:
                splits = self.get_split_history(symbol)
                if len(splits) > 0:
                    self.backfill_split_adjusted_prices(symbol)
                    adjusted_count += 1
                else:
                    print(f"No splits for {symbol}")
            except Exception as e:
                print(f"Error adjusting {symbol}: {e}")

        print(f"✓ Completed split adjustments for {adjusted_count}/{len(symbols)} symbols")

    def store_api_data(self, symbol: str, df: pd.DataFrame, source: str = "api") -> None:
        """
        Store API-fetched data in the database with proper metadata.

        Args:
            symbol (str): Stock symbol
            df (pd.DataFrame): Raw API data (without metadata columns)
            source (str): Data source identifier. Defaults to "api"
        """
        # Add metadata columns required for database format
        df_with_metadata = df.copy()
        df_with_metadata['symbol'] = symbol.upper()
        df_with_metadata['source'] = source.lower()
        df_with_metadata['last_updated'] = datetime.now()

        # Update database
        self.update_data(df_with_metadata)

    def get_cached_data(self, symbol: str, start_date: str, end_date: str, source: str = None) -> Optional[pd.DataFrame]:
        """
        Get cached data for the specified parameters, returning only price columns.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            source (str, optional): Data source filter

        Returns:
            pd.DataFrame or None: Price data without metadata columns
        """
        df = self.query(symbol=symbol, source=source, start_date=start_date, end_date=end_date)
        if df is None or len(df) == 0:
            return None

        # Remove metadata columns and return only price data
        metadata_cols = ['symbol', 'source', 'last_updated']
        price_cols = [col for col in df.columns if col not in metadata_cols]

        return df[price_cols] if price_cols else None

    def backfill_sparse_prices(self, symbol: str, columns: Optional[List[str]] = None, freq: str = 'D') -> None:
        """
        Backfill sparse datetime and price data for a symbol by creating a complete date range
        and forward-filling with the latest previous values.
        Updates the loaded data in the Database.

        Args:
            symbol (str): Stock symbol to backfill
            columns (List[str], optional): Specific columns to backfill. If None, backfills all price columns.
            freq (str): Frequency for date range ('D' for daily, 'B' for business days). Default is 'D'.

        Example:
            Given sparse dates [2023-01-01, 2023-01-05, 2023-01-10] with values [a, b, c],
            becomes complete daily range with forward-filled values:
            [2023-01-01: a, 2023-01-02: a, 2023-01-03: a, 2023-01-04: a, 2023-01-05: b, ...]

        Usage:
            >>> database.backfill_sparse_prices('AAPL')  # Daily frequency
            >>> database.backfill_sparse_prices('AAPL', freq='B')  # Business days only
        """
        symbol_data = self.query(symbol=symbol)

        if symbol_data is None or len(symbol_data) == 0:
            print(f"No data found for symbol {symbol}")
            return

        # Determine which columns to backfill
        if columns is None:
            # Default to all numeric columns that typically contain price data
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close']
            columns = [col for col in price_columns if col in symbol_data.columns]

            # If no standard price columns found, use all numeric columns
            if not columns:
                columns = symbol_data.select_dtypes(include=[np.number]).columns.tolist()

        available_columns = [col for col in columns if col in symbol_data.columns]

        if not available_columns:
            print(f"No columns to backfill for {symbol}")
            return

        # Create complete date range
        start_date = symbol_data.index.min()
        end_date = symbol_data.index.max()

        print(f"Creating complete date range for {symbol} from {start_date.date()} to {end_date.date()} (freq: {freq})")

        if freq == 'B':
            # Business days only (excludes weekends)
            complete_dates = pd.bdate_range(start=start_date, end=end_date, freq='D')
        else:
            # All days
            complete_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Create DataFrame with complete date range
        filled_data = symbol_data.reindex(complete_dates)

        print(f"Backfilling {symbol} on columns: {available_columns}")
        print(f"Original data points: {len(symbol_data)}, Complete range: {len(filled_data)}")

        # Forward fill the specified columns
        for col in available_columns:
            if col in filled_data.columns:
                filled_data[col] = filled_data[col].fillna(method='ffill')

        # Forward fill metadata columns if they exist
        metadata_columns = ['symbol', 'source', 'last_updated']
        for col in metadata_columns:
            if col in filled_data.columns:
                filled_data[col] = filled_data[col].fillna(method='ffill')

        # Report statistics
        original_count = len(symbol_data)
        filled_count = len(filled_data)
        new_dates_added = filled_count - original_count

        print(f"✓ Added {new_dates_added} missing dates")

        # Check for remaining nulls in price columns
        remaining_nulls = filled_data[available_columns].isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"⚠️  {remaining_nulls} null values remain in price columns (no previous value to fill from)")

        # Update the internal DataFrame
        symbol_mask = self._df['symbol'] == symbol.upper()

        # Remove old data for this symbol
        self._df = self._df[~symbol_mask]

        # Add backfilled data
        self._df = pd.concat([self._df, filled_data])
        self._df = self._df.sort_index()

        # Save to file
        self.save_dataframe(self._df)

        print(f"✓ Backfilled sparse dates and prices saved for {symbol}")

    def delete_by_query(self, symbol: str = None, source: str = None, start_date: str = None, end_date: str = None) -> int:
        """
        Delete records from the database based on query parameters.

        Args:
            symbol (str, optional): Stock symbol to delete
            source (str, optional): Data source to delete
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format

        Returns:
            int: Number of records deleted

        Example:
            >>> # Delete all data for a specific symbol
            >>> database.delete_by_query(symbol='AAPL')

            >>> # Delete data for a symbol within a date range
            >>> database.delete_by_query(symbol='AAPL', start_date='2020-01-01', end_date='2020-12-31')

            >>> # Delete all data from a specific source
            >>> database.delete_by_query(source='cpi_inflation')
        """
        if self._df is None:
            self._load_data()

        if self._df is None or len(self._df) == 0:
            print("No data in database to delete")
            return 0

        # Start with all data
        mask = pd.Series([True] * len(self._df), index=self._df.index)

        # Apply symbol filter
        if symbol is not None:
            symbol_mask = self._df['symbol'] == symbol.upper()
            mask = mask & symbol_mask
            print(f"Filtering by symbol: {symbol.upper()}")

        # Apply source filter
        if source is not None:
            if 'source' in self._df.columns:
                source_mask = self._df['source'] == source
                mask = mask & source_mask
                print(f"Filtering by source: {source}")
            else:
                print("Warning: 'source' column not found in database")

        # Apply date range filter
        if start_date is not None or end_date is not None:
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                date_mask = self._df.index >= start_dt
                mask = mask & date_mask
                print(f"Filtering from date: {start_date}")

            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                date_mask = self._df.index <= end_dt
                mask = mask & date_mask
                print(f"Filtering to date: {end_date}")

        # Count records to be deleted
        records_to_delete = mask.sum()

        if records_to_delete == 0:
            print("No records match the deletion criteria")
            return 0

        # Show what will be deleted
        print(f"Found {records_to_delete} records matching deletion criteria")

        # Delete the records (keep everything that doesn't match the mask)
        self._df = self._df[~mask]

        # Save the updated data
        self.save_dataframe(self._df)

        print(f"✓ Deleted {records_to_delete} records from database")
        return records_to_delete
