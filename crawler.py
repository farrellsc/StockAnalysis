import pandas as pd
import pickle  # Still needed for temporary database creation
import os
import time
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from database import Database
from api_source import ApiSource, TiingoApiSource, AshareApiSource
from structs import BASE_DIR, DATA_DIR, PORTFOLIO_DIR


class Crawler:
    """
    A flexible class for crawling end-of-day stock data from various sources.

    This crawler supports multiple data sources through a pluggable architecture
    and saves all data in a single consolidated DataFrame with proper indexing.
    """

    def __init__(self, api_source: Optional[ApiSource] = None, data_dir: str = None,
                 consolidated_file: str = "stock_data.pkl", **source_config):
        """
        Initialize the Crawler with a configurable data source.

        Args:
            api_source (ApiSource, optional): Data source instance. If None, defaults to TiingoApiSource
            data_dir (str): Directory to save pickle files. If None, uses DATA_DIR from structs
            consolidated_file (str): Name of the consolidated pickle file. Defaults to "stock_data.pkl"
            **source_config: Configuration passed to data source if creating default
        """
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.consolidated_file_path = os.path.join(self.data_dir, consolidated_file)
        self._database = None  # Lazy initialization

        # Use provided data source or create default Tiingo source
        if api_source is not None:
            self.api_source = api_source
            source_name = api_source.__class__.__name__
        else:
            self.api_source = TiingoApiSource(**source_config)
            source_name = "TiingoApiSource"

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        print(f"✓ Crawler initialized with {source_name}")
        print(f"✓ Data will be stored in: {self.consolidated_file_path}")

    def _get_database(self) -> Optional[Database]:
        """Get database instance with lazy initialization. Returns None if file doesn't exist."""
        if not os.path.exists(self.consolidated_file_path):
            return None

        if self._database is None:
            try:
                self._database = Database(self.consolidated_file_path)
            except Exception:
                return None
        else:
            # Refresh database to get latest data
            self._database.refresh()

        return self._database

    def crawl_single(self, symbol: str, start_date: str, end_date: str, force: bool = False) -> str:
        """
        Crawl end-of-day stock data and add to consolidated DataFrame.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            force (bool): If True, force re-crawling even if data exists. Defaults to False.

        Returns:
            str: Path to the consolidated pickle file

        Raises:
            ValueError: If input parameters are invalid
            Exception: If API request or file operations fail
        """
        # Input validation
        self._validate_inputs(symbol, start_date, end_date)

        # Check if data already exists (unless force is True)
        if not force and self._data_exists(symbol, start_date, end_date):
            print(f"✓ Data already exists for {symbol.upper()} from {start_date} to {end_date}")
            print(f"📂 Skipping crawl. Use force=True to re-crawl existing data.")
            return self.consolidated_file_path

        # Fetch data using the configured data source (quota management handled by source)
        raw_df = self.api_source.fetch_data(symbol, start_date, end_date)

        # Add metadata columns for partitioning
        df_with_metadata = self._add_metadata(raw_df, symbol)

        # Load existing consolidated data and merge
        self._merge_and_save_data(df_with_metadata)

        return self.consolidated_file_path

    def crawl(self, symbols: List[str], start_date: str, end_date: str, force: bool = False,
              batch_size: int = 10) -> str:
        """
        Crawl multiple symbols efficiently with batched database write operations.

        Args:
            symbols (List[str]): List of stock symbols to crawl
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            force (bool): If True, force re-crawling even if data exists. Defaults to False.
            batch_size (int): Number of symbols to process before writing to database. Defaults to 10.

        Returns:
            str: Path to the consolidated pickle file

        Raises:
            ValueError: If input parameters are invalid
            Exception: If API requests or file operations fail
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        print(f"🚀 Starting batch crawl for {len(symbols)} symbols from {start_date} to {end_date}")
        print(f"📦 Using batch size of {batch_size} symbols per database write")

        # Track overall progress
        all_successful_symbols = []
        all_failed_symbols = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        # Process symbols in batches
        for batch_num in range(0, len(symbols), batch_size):
            batch_symbols = symbols[batch_num:batch_num + batch_size]
            current_batch = (batch_num // batch_size) + 1

            print(f"\n📦 Processing batch {current_batch}/{total_batches} ({len(batch_symbols)} symbols)")

            # Collect DataFrames for this batch
            batch_dataframes = []
            batch_successful = []
            batch_failed = []

            for i, symbol in enumerate(batch_symbols):
                symbol_index = batch_num + i + 1
                try:
                    print(f"[{symbol_index}/{len(symbols)}] Processing {symbol.upper()}...")

                    # Input validation for this symbol
                    self._validate_inputs(symbol, start_date, end_date)

                    # Check if data already exists (unless force is True)
                    if not force and self._data_exists(symbol, start_date, end_date):
                        print(f"✓ Data already exists for {symbol.upper()}, skipping")
                        continue

                    # Fetch data using the configured data source (quota management handled by source)
                    raw_df = self.api_source.fetch_data(symbol, start_date, end_date)

                    # Add metadata columns for partitioning
                    df_with_metadata = self._add_metadata(raw_df, symbol)
                    batch_dataframes.append(df_with_metadata)
                    batch_successful.append(symbol.upper())

                except Exception as e:
                    print(f"❌ Failed to crawl {symbol.upper()}: {e}")
                    batch_failed.append(symbol.upper())
                    continue

            # Write this batch to database if we have data
            if batch_dataframes:
                try:
                    print(f"💾 Writing batch {current_batch} ({len(batch_dataframes)} symbols) to database...")

                    # Combine batch DataFrames
                    batch_combined_df = pd.concat(batch_dataframes, ignore_index=False)
                    batch_combined_df = batch_combined_df.sort_index()

                    # Load existing consolidated data and merge
                    self._merge_and_save_data(batch_combined_df)

                    print(f"✅ Successfully saved batch {current_batch}")
                    all_successful_symbols.extend(batch_successful)

                except Exception as e:
                    print(f"❌ Failed to save batch {current_batch}: {e}")
                    # Mark all symbols in this batch as failed
                    all_failed_symbols.extend(batch_successful)
                    all_failed_symbols.extend(batch_failed)
                    continue

            # Add failed symbols from this batch
            all_failed_symbols.extend(batch_failed)

        # Final Summary
        print(f"\n📊 Final Crawl Summary:")
        print(f"✅ Total Successful: {len(all_successful_symbols)} symbols")
        if all_successful_symbols:
            # Show first 10 symbols, then count if more
            display_symbols = all_successful_symbols[:10]
            remaining_count = len(all_successful_symbols) - 10
            if remaining_count > 0:
                print(f"   {', '.join(display_symbols)} (and {remaining_count} more)")
            else:
                print(f"   {', '.join(display_symbols)}")

        if all_failed_symbols:
            print(f"❌ Total Failed: {len(all_failed_symbols)} symbols")
            display_failed = all_failed_symbols[:10]
            remaining_failed = len(all_failed_symbols) - 10
            if remaining_failed > 0:
                print(f"   {', '.join(display_failed)} (and {remaining_failed} more)")
            else:
                print(f"   {', '.join(display_failed)}")

        return self.consolidated_file_path

    def set_api_source(self, api_source: ApiSource) -> None:
        """
        Change the data source used by the crawler.

        Args:
            api_source (ApiSource): New data source instance
        """
        self.api_source = api_source
        print(f"✓ Data source changed to {api_source.__class__.__name__}")

    def get_supported_sources(self) -> Dict[str, type]:
        """
        Get a dictionary of available data source types.

        Returns:
            Dict[str, type]: Mapping of source names to their classes
        """
        return {
            'tiingo': TiingoApiSource,
            'ashare': AshareApiSource,
            # Future data sources can be added here
            # 'alpha_vantage': AlphaVantageApiSource,
            # 'yahoo_finance': YahooFinanceApiSource,
        }


    def _validate_inputs(self, symbol: str, start_date: str, end_date: str) -> None:
        """Validate input parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        if not start_date or not isinstance(start_date, str):
            raise ValueError("Start date must be a non-empty string")

        if not end_date or not isinstance(end_date, str):
            raise ValueError("End date must be a non-empty string")

        # Validate date formats
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")

        if start_dt > end_dt:
            raise ValueError("Start date must be before or equal to end date")

        # Additional validation using data source
        if not self.api_source.validate_symbol(symbol):
            raise ValueError(f"Symbol '{symbol}' is not valid for {self.api_source.__class__.__name__}")

    def _data_exists(self, symbol: str, start_date: str, end_date: str) -> bool:
        """
        Check if data already exists for the specified symbol and date range.

        Args:
            symbol (str): Stock symbol to check
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            bool: True if sufficient data exists for the range, False otherwise
        """
        database = self._get_database()
        if database is None:
            return False

        # Query existing data for the specific symbol and source
        source_name = self._get_source_name()
        symbol_data = database.query(symbol=symbol, source=source_name)

        if len(symbol_data) == 0:
            return False

        # Check if we have complete coverage for the requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get the date range from existing data (convert to date strings)
        existing_dates = set(symbol_data.index.strftime('%Y-%m-%d'))

        # Generate all business days in the requested range (convert to date strings)
        requested_dates = set(pd.bdate_range(start_dt, end_dt).strftime('%Y-%m-%d'))

        # Check if we have sufficient data coverage (>= 90%)
        missing_dates = requested_dates - existing_dates
        coverage_ratio = (len(requested_dates) - len(missing_dates)) / len(requested_dates)

        if coverage_ratio >= 0.9:  # 90% threshold
            if len(missing_dates) == 0:
                print(f"📊 Found complete data for {symbol.upper()} ({source_name}): {len(existing_dates)} days")
            else:
                print(f"📊 Found sufficient data for {symbol.upper()} ({source_name}): "
                      f"{coverage_ratio:.1%} coverage ({len(existing_dates)} days, missing {len(missing_dates)} days)")
            return True
        else:
            print(f"📊 Insufficient data for {symbol.upper()} ({source_name}): "
                  f"{coverage_ratio:.1%} coverage ({len(existing_dates)} days, missing {len(missing_dates)} days)")
            return False

    def _add_metadata(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add metadata columns for partitioning and identification."""
        df_copy = df.copy()

        # Add partitioning columns
        df_copy['symbol'] = symbol.upper()
        df_copy['source'] = self._get_source_name()
        df_copy['last_updated'] = datetime.now()

        return df_copy

    def _get_source_name(self) -> str:
        """Get a clean source name for partitioning."""
        source_class = self.api_source.__class__.__name__
        if 'tiingo' in source_class.lower():
            return 'tiingo'
        elif 'ashare' in source_class.lower():
            return 'ashare'
        # Add other source mappings here as needed
        return source_class.lower()

    def _merge_and_save_data(self, new_df: pd.DataFrame) -> None:
        """Merge new data with existing consolidated data and save using Database API."""
        database = self._get_database()

        if database is not None:
            # Use Database's update_data method for smart merging
            database.update_data(new_df)
        else:
            # Create new database with the data
            self._create_database_with_data(new_df)

    def _create_database_with_data(self, df: pd.DataFrame) -> None:
        """Create a new database file with the provided data."""
        try:
            # Use Database save functionality by creating a minimal database instance
            from database import Database

            # Create the file first with a temporary database
            with open(self.consolidated_file_path, 'wb') as f:
                pickle.dump(df, f)

            # Now create database instance and let it manage the data
            database = Database(self.consolidated_file_path)
            print(f"📊 Created new dataset with {len(df)} rows")

            # Update our internal reference
            self._database = database

        except Exception as e:
            raise Exception(f"Failed to create database: {e}")

    def backfill_portfolio_stocks(self, portfolio_names: List[str], force: bool = False) -> Dict[str, List[str]]:
        """
        Fetch all data for the symbols until the biggest date defined in portfolio.

        Args:
            portfolio_names (List[str]): List of portfolio file names (without .json extension)
            force (bool): Whether to force re-download existing data

        Returns:
            Dict[str, List[str]]: Summary of successful and failed symbols
        """
        import json

        print(f"🔍 Processing {len(portfolio_names)} portfolio file(s)")

        # Build full file paths and validate
        portfolio_files = []
        for name in portfolio_names:
            portfolio_file = os.path.join(PORTFOLIO_DIR, f"{name}.json")
            if os.path.exists(portfolio_file):
                portfolio_files.append(portfolio_file)
                print(f"✓ Found: {name}.json")
            else:
                print(f"⚠️  Not found: {name}.json")

        if not portfolio_files:
            print(f"❌ No valid portfolio files found")
            return {"successful": [], "failed": []}

        # Extract all stock symbols and find the biggest (latest) date
        all_symbols = set()
        biggest_date = None

        for portfolio_file in portfolio_files:
            try:
                print(f"📄 Processing: {os.path.basename(portfolio_file)}")

                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)

                # Find the biggest date and extract all symbols
                for date_str, compositions in portfolio_data.items():
                    try:
                        # Parse the date
                        portfolio_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

                        # Track the biggest (latest) date
                        if biggest_date is None or portfolio_date > biggest_date:
                            biggest_date = portfolio_date

                        # Extract stock symbols from compositions
                        if isinstance(compositions, dict):
                            for symbol in compositions.keys():
                                if isinstance(symbol, str) and symbol.upper() not in ['CASH', 'TOTAL']:
                                    all_symbols.add(symbol.upper())

                    except ValueError as e:
                        print(f"⚠️  Skipping invalid date format in {portfolio_file}: {date_str}")
                        continue

            except Exception as e:
                print(f"❌ Error processing {portfolio_file}: {e}")
                continue

        if not all_symbols:
            print("📝 No stock symbols found in portfolio files")
            return {"successful": [], "failed": []}

        if biggest_date is None:
            print("❌ No valid dates found in portfolio files")
            return {"successful": [], "failed": list(all_symbols)}

        # Convert biggest date to string format
        end_date = biggest_date.strftime("%Y-%m-%d")

        print(f"🎯 Found {len(all_symbols)} unique stock symbols")
        print(f"📅 Biggest date in portfolio: {end_date}")

        # Query database to find the actual start date for each symbol
        database = self._get_database()
        symbols_list = sorted(list(all_symbols))

        if database is not None:
            # Find the earliest start date needed across all symbols
            earliest_start_date = None

            for symbol in symbols_list:
                try:
                    # Get the date range for this symbol in existing data
                    date_range = database.get_date_range(symbol=symbol)
                    existing_end_date = date_range.get('end_date')

                    if existing_end_date:
                        # Parse existing end date and add 1 day for backfill start
                        existing_end = datetime.strptime(existing_end_date, "%Y-%m-%d")
                        symbol_start_date = (existing_end + timedelta(days=1)).strftime("%Y-%m-%d")
                        print(f"   {symbol}: existing data until {existing_end_date}, will fetch from {symbol_start_date}")
                    else:
                        # No existing data, use 2 years before biggest date as fallback
                        symbol_start_date = (biggest_date - timedelta(days=730)).strftime("%Y-%m-%d")
                        print(f"   {symbol}: no existing data, will fetch from {symbol_start_date}")

                    # Track the earliest start date needed
                    if earliest_start_date is None or symbol_start_date < earliest_start_date:
                        earliest_start_date = symbol_start_date

                except Exception as e:
                    # Error checking existing data, use fallback
                    symbol_start_date = (biggest_date - timedelta(days=730)).strftime("%Y-%m-%d")
                    print(f"   {symbol}: error checking existing data, will fetch from {symbol_start_date}")

                    if earliest_start_date is None or symbol_start_date < earliest_start_date:
                        earliest_start_date = symbol_start_date

            start_date = earliest_start_date
        else:
            # No database yet, use 2 years before biggest date as fallback
            start_date = (biggest_date - timedelta(days=730)).strftime("%Y-%m-%d")
            print("📊 No existing database found, using 2-year lookback")

        print(f"📊 Fetching data from {start_date} to {end_date}")

        try:
            # Crawl all symbols from start_date to biggest_date
            result_path = self.crawl(
                symbols=symbols_list,
                start_date=start_date,
                end_date=end_date,
                force=force
            )

            print(f"✅ Portfolio stock data backfill completed")
            print(f"📁 Data saved to: {result_path}")

            # Return success - the crawl method handles detailed success/failure reporting
            return {"successful": symbols_list, "failed": []}

        except Exception as e:
            print(f"❌ Error during data crawling: {e}")
            return {"successful": [], "failed": symbols_list}

    def get_quota_status(self) -> Optional[Dict]:
        """Get current quota status information from the API source."""
        return self.api_source.get_quota_status()
