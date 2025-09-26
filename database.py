import pandas as pd
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
              source: Optional[str] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Query data with flexible filtering options.

        Args:
            symbol (str or List[str], optional): Symbol(s) to filter by
            source (str, optional): Data source to filter by
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
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
        Removes duplicates for same symbol/source/date combinations.

        Args:
            new_df (pd.DataFrame): New data to merge

        Raises:
            Exception: If update operation fails
        """
        try:
            # Validate new data format
            self._validate_dataframe(new_df)

            if len(self._df) > 0:
                # Remove overlapping data to avoid duplicates
                for _, new_row in new_df.iterrows():
                    symbol = new_row['symbol']
                    source = new_row['source']
                    date = new_row.name  # Index value (date)

                    # Remove existing data for same symbol/source/date
                    mask = ~((self._df['symbol'] == symbol) &
                             (self._df['source'] == source) &
                             (self._df.index == date))
                    self._df = self._df[mask]

                # Combine with new data
                consolidated_df = pd.concat([self._df, new_df])
                print(f"📊 Merged data: {len(self._df)} existing + {len(new_df)} new = {len(consolidated_df)} total rows")
            else:
                consolidated_df = new_df
                print(f"📊 Created new dataset with {len(consolidated_df)} rows")

            # Sort and save
            consolidated_df = consolidated_df.sort_index()
            self.save_dataframe(consolidated_df)

        except Exception as e:
            raise Exception(f"Failed to update database: {e}")

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

        # Find dates where split factor > 1
        splits = symbol_data[symbol_data[split_col] > 1.0].copy()

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
