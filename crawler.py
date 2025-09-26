import requests
import pandas as pd
import pickle  # Still needed for temporary database creation
import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod
from database import Database


class QuotaManager:
    """
    Manages hourly API quotas with persistent tracking.
    """

    def __init__(self, quota_file: str = "data/.quota_tracker.json"):
        """
        Initialize quota manager.

        Args:
            quota_file (str): Path to quota tracking file
        """
        self.quota_file = quota_file
        self._ensure_quota_dir()

    def _ensure_quota_dir(self):
        """Ensure quota file directory exists."""
        os.makedirs(os.path.dirname(self.quota_file), exist_ok=True)

    def _get_current_hour_key(self) -> str:
        """Get current hour as a string key."""
        return datetime.now().strftime('%Y-%m-%d-%H')

    def _load_quota_data(self) -> Dict:
        """Load quota tracking data from file."""
        if not os.path.exists(self.quota_file):
            return {}

        try:
            with open(self.quota_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_quota_data(self, data: Dict):
        """Save quota tracking data to file."""
        try:
            with open(self.quota_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail if can't save

    def _cleanup_old_entries(self, data: Dict) -> Dict:
        """Remove entries older than 24 hours."""
        current_time = datetime.now()
        cleaned_data = {}

        for hour_key, count in data.items():
            try:
                hour_time = datetime.strptime(hour_key, '%Y-%m-%d-%H')
                if current_time - hour_time < timedelta(hours=24):
                    cleaned_data[hour_key] = count
            except ValueError:
                continue  # Skip invalid entries

        return cleaned_data

    def get_current_usage(self) -> int:
        """Get current hour's API usage count."""
        data = self._load_quota_data()
        current_hour = self._get_current_hour_key()
        return data.get(current_hour, 0)

    def can_make_request(self, quota_limit: int) -> bool:
        """Check if we can make another API request within quota."""
        return self.get_current_usage() < quota_limit

    def record_request(self):
        """Record that an API request was made."""
        data = self._load_quota_data()
        data = self._cleanup_old_entries(data)

        current_hour = self._get_current_hour_key()
        data[current_hour] = data.get(current_hour, 0) + 1

        self._save_quota_data(data)

    def time_until_next_hour(self) -> int:
        """Get seconds until the next hour starts."""
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return int((next_hour - now).total_seconds())

    def get_quota_status(self, quota_limit: int) -> Dict:
        """Get current quota status information."""
        current_usage = self.get_current_usage()
        return {
            'current_hour': self._get_current_hour_key(),
            'requests_made': current_usage,
            'quota_limit': quota_limit,
            'requests_remaining': max(0, quota_limit - current_usage),
            'quota_full': current_usage >= quota_limit,
            'seconds_until_reset': self.time_until_next_hour()
        }


class CrawlSource(ABC):
    """
    Abstract base class for stock data sources.

    This class defines the interface that all data sources must implement
    to be compatible with the Crawler.
    """

    @abstractmethod
    def __init__(self, **config):
        """Initialize the data source with configuration."""
        pass

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data for the given symbol and date range.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Stock data with standardized columns
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if the symbol is supported by this data source."""
        pass


class TiingoCrawlSource(CrawlSource):
    """Tiingo API data source implementation with quota management."""

    def __init__(self, api_key: Optional[str] = None, quota_manager: Optional[QuotaManager] = None, **config):
        """
        Initialize Tiingo data source.

        Args:
            api_key (str, optional): Tiingo API key
            quota_manager (QuotaManager, optional): Quota manager instance
            **config: Additional configuration options
        """
        self.api_key = self._get_api_key(api_key)
        self.base_url = config.get('base_url', 'https://api.tiingo.com/tiingo/daily')
        self.quota_manager = quota_manager

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }

    def _get_api_key(self, provided_key: Optional[str]) -> str:
        """Get Tiingo API key from various sources."""
        if provided_key:
            return provided_key

        env_key = os.getenv('TIINGO_API_KEY')
        if env_key:
            return env_key

        try:
            from cert import TiingoKey
            return TiingoKey
        except ImportError:
            pass

        raise ValueError(
            "Tiingo API key not found. Provide it during initialization, "
            "set TIINGO_API_KEY environment variable, or ensure TiingoKey is defined in cert.py"
        )

    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format for Tiingo (basic validation)."""
        return isinstance(symbol, str) and len(symbol.strip()) > 0

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Tiingo API."""
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid symbol: {symbol}")

        url = f"{self.base_url}/{symbol.upper()}/prices"
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }

        try:
            print(f"📡 [Tiingo] Fetching data for {symbol.upper()} from {start_date} to {end_date}...")

            response = requests.get(url, headers=self.headers, params=params)

            # Record the API request for quota tracking
            if self.quota_manager:
                self.quota_manager.record_request()

            response.raise_for_status()

            data = response.json()

            if not data:
                raise ValueError(f"No data returned for symbol {symbol.upper()}")

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Standardize column names for consistency across data sources
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjOpen': 'Adj_Open',
                'adjHigh': 'Adj_High',
                'adjLow': 'Adj_Low',
                'adjClose': 'Adj_Close',
                'adjVolume': 'Adj_Volume',
                'divCash': 'Dividend',
                'splitFactor': 'Split_Factor'
            })

            print(f"✓ [Tiingo] Successfully fetched {len(df)} days of data")
            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Tiingo API request failed: {e}")
        except Exception as e:
            raise Exception(f"Error fetching Tiingo data: {e}")


class Crawler:
    """
    A flexible class for crawling end-of-day stock data from various sources.

    This crawler supports multiple data sources through a pluggable architecture
    and saves all data in a single consolidated DataFrame with proper indexing.
    """

    def __init__(self, crawl_source: Optional[CrawlSource] = None, data_dir: str = "data",
                 consolidated_file: str = "stock_data.pkl", quota_limit: Optional[int] = None, **source_config):
        """
        Initialize the Crawler with a configurable data source and quota management.

        Args:
            crawl_source (CrawlSource, optional): Data source instance. If None, defaults to TiingoCrawlSource
            data_dir (str): Directory to save pickle files. Defaults to "data"
            consolidated_file (str): Name of the consolidated pickle file. Defaults to "stock_data.pkl"
            quota_limit (int, optional): Hourly API quota limit. If None, no quota management
            **source_config: Configuration passed to data source if creating default
        """
        self.data_dir = data_dir
        self.consolidated_file_path = os.path.join(data_dir, consolidated_file)
        self._database = None  # Lazy initialization
        self.quota_limit = quota_limit

        # Initialize quota manager if quota limit is specified
        if quota_limit is not None:
            self.quota_manager = QuotaManager(quota_file=os.path.join(data_dir, ".quota_tracker.json"))
        else:
            self.quota_manager = None

        # Use provided data source or create default Tiingo source
        if crawl_source is not None:
            self.crawl_source = crawl_source
            source_name = crawl_source.__class__.__name__
        else:
            self.crawl_source = TiingoCrawlSource(quota_manager=self.quota_manager, **source_config)
            source_name = "TiingoCrawlSource"

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

        # Check quota before making API request
        if self.quota_manager and self.quota_limit:
            self._wait_for_quota_if_needed()

        # Fetch data using the configured data source
        raw_df = self.crawl_source.fetch_data(symbol, start_date, end_date)

        # Add metadata columns for partitioning
        df_with_metadata = self._add_metadata(raw_df, symbol)

        # Load existing consolidated data and merge
        self._merge_and_save_data(df_with_metadata)

        return self.consolidated_file_path

    def crawl(self, symbols: List[str], start_date: str, end_date: str, force: bool = False, batch_size: int = 10) -> str:
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

                    # Check quota before making API request
                    if self.quota_manager and self.quota_limit:
                        self._wait_for_quota_if_needed()

                    # Fetch data using the configured data source
                    raw_df = self.crawl_source.fetch_data(symbol, start_date, end_date)

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

    def set_crawl_source(self, crawl_source: CrawlSource) -> None:
        """
        Change the data source used by the crawler.

        Args:
            crawl_source (CrawlSource): New data source instance
        """
        self.crawl_source = crawl_source
        print(f"✓ Data source changed to {crawl_source.__class__.__name__}")

    def get_supported_sources(self) -> Dict[str, type]:
        """
        Get a dictionary of available data source types.

        Returns:
            Dict[str, type]: Mapping of source names to their classes
        """
        return {
            'tiingo': TiingoCrawlSource,
            # Future data sources can be added here
            # 'alpha_vantage': AlphaVantageCrawlSource,
            # 'yahoo_finance': YahooFinanceCrawlSource,
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
        if not self.crawl_source.validate_symbol(symbol):
            raise ValueError(f"Symbol '{symbol}' is not valid for {self.crawl_source.__class__.__name__}")

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
        source_class = self.crawl_source.__class__.__name__
        if 'tiingo' in source_class.lower():
            return 'tiingo'
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

    def _wait_for_quota_if_needed(self):
        """Wait until quota allows for the next request."""
        if not self.quota_manager.can_make_request(self.quota_limit):
            status = self.quota_manager.get_quota_status(self.quota_limit)
            wait_seconds = status['seconds_until_reset']

            print(f"⏳ Quota limit reached ({status['requests_made']}/{self.quota_limit} requests this hour)")
            print(f"⏰ Waiting {wait_seconds} seconds until next hour...")

            # Wait with progress updates every 60 seconds
            while wait_seconds > 0:
                if wait_seconds >= 60:
                    print(f"⏰ {wait_seconds} seconds remaining until quota reset...")
                    time.sleep(60)
                    wait_seconds -= 60
                else:
                    time.sleep(wait_seconds)
                    wait_seconds = 0

            print("✓ Quota reset - resuming crawling")

    def get_quota_status(self) -> Optional[Dict]:
        """Get current quota status information."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.get_quota_status(self.quota_limit)
        return None