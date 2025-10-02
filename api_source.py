import requests
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, List
from abc import ABC, abstractmethod
import logging


class QuotaManager:
    """
    Manages hourly API quotas with persistent tracking.
    """

    def __init__(self, quota_file: str = "data/.quota_tracker.json", log_level: str = 'INFO'):
        """
        Initialize quota manager.

        Args:
            quota_file (str): Path to quota tracking file
            log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        self.quota_file = quota_file

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._ensure_quota_dir()
        self.logger.debug(f"QuotaManager initialized with quota file: {quota_file}")

    def _ensure_quota_dir(self):
        """Ensure quota file directory exists."""
        os.makedirs(os.path.dirname(self.quota_file), exist_ok=True)

    def _get_current_hour_key(self) -> str:
        """Get current hour as a string key."""
        return datetime.now().strftime('%Y-%m-%d-%H')

    def _load_quota_data(self) -> dict:
        """Load quota tracking data from file."""
        if not os.path.exists(self.quota_file):
            return {}

        try:
            with open(self.quota_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_quota_data(self, data: dict):
        """Save quota tracking data to file."""
        try:
            with open(self.quota_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail if can't save

    def _cleanup_old_entries(self, data: dict) -> dict:
        """Remove entries older than 24 hours."""
        from datetime import timedelta

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
        from datetime import timedelta

        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return int((next_hour - now).total_seconds())

    def get_quota_status(self, quota_limit: int) -> dict:
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


class ApiSource(ABC):
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



class TiingoApiSource(ApiSource):
    """Tiingo API data source implementation with quota management."""

    def __init__(self, api_key: Optional[str] = None, quota_limit: Optional[int] = None, log_level: str = 'INFO', **config):
        """
        Initialize Tiingo data source.

        Args:
            api_key (str, optional): Tiingo API key
            quota_limit (int, optional): Hourly quota limit for Tiingo API
            log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            **config: Additional configuration options
        """
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Create handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.api_key = self._get_api_key(api_key)
        self.base_url = config.get('base_url', 'https://api.tiingo.com/tiingo/daily')
        self.quota_limit = quota_limit

        self.logger.info(f"TiingoApiSource initialized with quota_limit: {quota_limit}")

        # Initialize quota manager if quota limit is specified
        if quota_limit is not None:
            quota_file = config.get('quota_file', "data/.tiingo_quota.json")
            self.quota_manager = QuotaManager(quota_file=quota_file)
        else:
            self.quota_manager = None

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

        # Check quota before making request
        self.wait_for_quota_if_needed()

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

    def can_make_request(self) -> bool:
        """Check if we can make another API request within quota."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.can_make_request(self.quota_limit)
        return True  # No quota limit set

    def get_quota_status(self) -> Optional[dict]:
        """Get current quota status information."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.get_quota_status(self.quota_limit)
        return None

    def wait_for_quota_if_needed(self) -> None:
        """Wait until quota allows for the next request."""
        if not self.can_make_request():
            import time
            status = self.get_quota_status()
            wait_seconds = status['seconds_until_reset']

            print(f"⏳ [Tiingo] Quota limit reached ({status['requests_made']}/{self.quota_limit} requests this hour)")
            print(f"⏰ [Tiingo] Waiting {wait_seconds} seconds until next hour...")

            # Wait with progress updates every 60 seconds
            while wait_seconds > 0:
                if wait_seconds >= 60:
                    print(f"⏰ [Tiingo] {wait_seconds} seconds remaining until quota reset...")
                    time.sleep(60)
                    wait_seconds -= 60
                else:
                    time.sleep(wait_seconds)
                    wait_seconds = 0

            print("✓ [Tiingo] Quota reset - resuming crawling")


class AshareApiSource(ApiSource):
    """Sina Finance API data source for Chinese A-share market data."""

    def __init__(self, quota_limit: Optional[int] = None, **config):
        """
        Initialize A-share data source.

        Args:
            quota_limit (int, optional): Hourly quota limit for Sina Finance API
            **config: Additional configuration options
        """
        self.base_url = config.get('base_url', 'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData')
        self.quota_limit = quota_limit
        self.frequency = config.get('frequency', '240')  # Daily data (240 minutes)
        self.ma_period = config.get('ma_period', 5)  # Moving average period

        # Initialize quota manager if quota limit is specified
        if quota_limit is not None:
            quota_file = config.get('quota_file', "data/.ashare_quota.json")
            self.quota_manager = QuotaManager(quota_file=quota_file)
        else:
            self.quota_manager = None

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate symbol format for A-share stocks.

        A-share symbols typically follow patterns like:
        - sh000001 (Shanghai index)
        - sz399001 (Shenzhen index)
        - sh600000 (Shanghai stock)
        - sz000001 (Shenzhen stock)
        """
        if not isinstance(symbol, str) or len(symbol.strip()) == 0:
            return False

        symbol = symbol.lower().strip()

        # Basic validation for A-share symbol format
        if len(symbol) >= 6:
            # Check for common A-share prefixes and numeric codes
            if (symbol.startswith(('sh', 'sz')) and
                symbol[2:].isdigit() and
                len(symbol[2:]) == 6):
                return True
            # Also accept pure numeric codes (will add prefix automatically)
            elif symbol.isdigit() and len(symbol) == 6:
                return True

        return False


    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch A-share data from Sina Finance API.

        Args:
            symbol (str): A-share symbol (e.g., 'sh600000', 'sz000001', or '600000')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Stock data with standardized columns
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Invalid A-share symbol: {symbol}")

        # Normalize symbol format
        normalized_symbol = self._normalize_symbol(symbol)

        # Calculate business days for accurate data length
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Calculate actual business days between start and end dates
        business_days = pd.bdate_range(start_dt, end_dt)
        data_length = len(business_days)

        # Add small buffer for holidays and ensure minimum request size
        data_length = max(data_length + 10, 20)

        try:
            print(f"📡 [Sina] Fetching A-share data for {normalized_symbol} from {start_date} to {end_date}...")

            # Build API URL
            url = f"{self.base_url}?symbol={normalized_symbol}&scale={self.frequency}&ma={self.ma_period}&datalen={data_length}"

            response = requests.get(url, timeout=30)

            # Record the API request for quota tracking
            if self.quota_manager:
                self.quota_manager.record_request()

            response.raise_for_status()

            # Parse JSON response
            data = json.loads(response.content.decode('utf-8'))

            if not data or not isinstance(data, list):
                raise ValueError(f"No data returned for symbol {normalized_symbol}")

            # Create DataFrame
            df = pd.DataFrame(data, columns=['day', 'open', 'high', 'low', 'close', 'volume'])

            # Convert data types
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Set up date index
            df['day'] = pd.to_datetime(df['day'])
            df.set_index('day', inplace=True)
            df.index.name = 'date'

            # Filter by date range - keep only requested business days
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Ensure we have data within the requested range
            if len(df) == 0:
                print(f"⚠️  [Sina] No data found for {normalized_symbol} in date range {start_date} to {end_date}")
            else:
                actual_start = df.index.min().strftime('%Y-%m-%d')
                actual_end = df.index.max().strftime('%Y-%m-%d')
                print(f"📊 [Sina] Retrieved data from {actual_start} to {actual_end} ({len(df)} days)")

            # Standardize column names to match other data sources
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Add required columns with default values (A-share data may not have these)
            df['Adj_Open'] = df['Open']
            df['Adj_High'] = df['High']
            df['Adj_Low'] = df['Low']
            df['Adj_Close'] = df['Close']
            df['Adj_Volume'] = df['Volume']
            df['Dividend'] = 0.0
            df['Split_Factor'] = 1.0

            # Remove rows with all NaN values
            df = df.dropna(how='all')

            print(f"✓ [Sina] Successfully fetched {len(df)} trading days of data")
            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Sina Finance API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Sina Finance response: {e}")
        except Exception as e:
            raise Exception(f"Error fetching Sina Finance data: {e}")

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Sina Finance API format.

        Args:
            symbol (str): Input symbol

        Returns:
            str: Normalized symbol (e.g., 'sh600000')
        """
        symbol = symbol.lower().strip()

        # If already has prefix, return as-is
        if symbol.startswith(('sh', 'sz')):
            return symbol

        # If numeric only, need to determine exchange
        if symbol.isdigit() and len(symbol) == 6:
            # Simple heuristic: 6xxxxx typically Shanghai, others Shenzhen
            if symbol.startswith('6'):
                return f'sh{symbol}'
            else:
                return f'sz{symbol}'

        return symbol

    def can_make_request(self) -> bool:
        """Check if we can make another API request within quota."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.can_make_request(self.quota_limit)
        return True  # No quota limit set

    def get_quota_status(self) -> Optional[dict]:
        """Get current quota status information."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.get_quota_status(self.quota_limit)
        return None

    def wait_for_quota_if_needed(self) -> None:
        """Wait until quota allows for the next request."""
        if not self.can_make_request():
            import time
            status = self.get_quota_status()
            wait_seconds = status['seconds_until_reset']

            print(f"⏳ [Sina] Quota limit reached ({status['requests_made']}/{self.quota_limit} requests this hour)")
            print(f"⏰ [Sina] Waiting {wait_seconds} seconds until next hour...")

            # Wait with progress updates every 60 seconds
            while wait_seconds > 0:
                if wait_seconds >= 60:
                    print(f"⏰ [Sina] {wait_seconds} seconds remaining until quota reset...")
                    time.sleep(60)
                    wait_seconds -= 60
                else:
                    time.sleep(wait_seconds)
                    wait_seconds = 0

            print("✓ [Sina] Quota reset - resuming crawling")


class BureauOfLaborStatisticsAPI(ApiSource):
    """Bureau of Labor Statistics API data source for economic indicators like CPI and unemployment."""

    def __init__(self, quota_limit: Optional[int] = None, **config):
        """
        Initialize Bureau of Labor Statistics data source.

        Args:
            quota_limit (int, optional): Hourly quota limit for BLS API
            **config: Additional configuration options
        """
        self.base_url = config.get('base_url', 'https://api.bls.gov/publicAPI/v2/timeseries/data/')
        self.quota_limit = quota_limit

        # Initialize quota manager if quota limit is specified
        if quota_limit is not None:
            quota_file = config.get('quota_file', "data/.bls_quota.json")
            self.quota_manager = QuotaManager(quota_file=quota_file)
        else:
            self.quota_manager = None

        # Define available datasets with BLS series IDs
        self.datasets = {
            'cpi_inflation': {
                'series_id': 'CUUR0000SA0',  # CPI-U: All items in U.S. city average, seasonally adjusted
                'description': 'Consumer Price Index (CPI) - Inflation Data',
                'value_field': 'cpi_value'
            },
            'unemployment_rate': {
                'series_id': 'LNS14000000',  # Unemployment rate: 16 years and over, seasonally adjusted
                'description': 'Unemployment Rate',
                'value_field': 'unemployment_rate'
            }
        }

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate symbol (dataset name) for BLS data.

        Args:
            symbol (str): Dataset name (e.g., 'cpi_inflation', 'unemployment_rate')

        Returns:
            bool: True if valid dataset name
        """
        return symbol.lower() in self.datasets

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch economic data from Bureau of Labor Statistics API.

        Args:
            symbol (str): Dataset name (e.g., 'cpi_inflation', 'unemployment_rate')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Economic data with standardized columns
        """
        if not self.validate_symbol(symbol):
            available = ', '.join(self.datasets.keys())
            raise ValueError(f"Invalid dataset: {symbol}. Available datasets: {available}")

        # Check quota before making request
        self.wait_for_quota_if_needed()

        dataset = self.datasets[symbol.lower()]

        try:
            # Convert dates to years for BLS API
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year

            print(f"📡 [BLS] Fetching {dataset['description']} from {start_year} to {end_year}...")

            # Prepare API request
            payload = {
                "seriesid": [dataset['series_id']],
                "startyear": str(start_year),
                "endyear": str(end_year),
                "registrationkey": ""  # Public API, no key needed
            }

            response = requests.post(self.base_url, json=payload, timeout=30)

            # Record the API request for quota tracking
            if self.quota_manager:
                self.quota_manager.record_request()

            response.raise_for_status()
            data = response.json()

            if data['status'] != 'REQUEST_SUCCEEDED':
                raise ValueError(f"BLS API request failed: {data.get('message', 'Unknown error')}")

            # Extract time series data
            series_data = data['Results']['series'][0]['data']

            # Convert to DataFrame
            records = []
            for item in series_data:
                # BLS returns data in format: year, period (M01-M12), value
                year = int(item['year'])
                period = item['period']
                value = float(item['value'])

                # Convert period to month (M01 = January, etc.)
                if period.startswith('M'):
                    month = int(period[1:])
                    # Create date as first day of the month
                    date_str = f"{year}-{month:02d}-01"
                    records.append({
                        'date': pd.to_datetime(date_str),
                        dataset['value_field']: value
                    })

            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            df = df.sort_index()

            # Filter by actual date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Create new DataFrame with only standardized columns matching AshareApiSource schema
            value_col = dataset['value_field']
            if value_col in df.columns:
                value_data = df[value_col]

                # Create new DataFrame with only standardized columns
                df_std = pd.DataFrame(index=df.index)

                # Map economic data to OHLCV columns (same value for all since it's a rate/index)
                df_std['Open'] = value_data
                df_std['High'] = value_data
                df_std['Low'] = value_data
                df_std['Close'] = value_data
                df_std['Volume'] = 0  # No volume for economic indicators

                # Add adjusted columns with same values (required by AshareApiSource schema)
                df_std['Adj_Open'] = value_data
                df_std['Adj_High'] = value_data
                df_std['Adj_Low'] = value_data
                df_std['Adj_Close'] = value_data
                df_std['Adj_Volume'] = 0
                df_std['Dividend'] = 0.0
                df_std['Split_Factor'] = 1.0

                print(f"✓ [BLS] Successfully fetched {len(df_std)} records")
                return df_std
            else:
                raise ValueError(f"Expected column {value_col} not found in BLS data")

        except requests.exceptions.RequestException as e:
            raise Exception(f"BLS API request failed: {e}")
        except Exception as e:
            raise Exception(f"Error fetching BLS data: {e}")

    def get_available_datasets(self) -> dict:
        """
        Get information about available datasets.

        Returns:
            dict: Dictionary with dataset names and descriptions
        """
        return {name: info['description'] for name, info in self.datasets.items()}

    def can_make_request(self) -> bool:
        """Check if we can make another API request within quota."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.can_make_request(self.quota_limit)
        return True  # No quota limit set

    def get_quota_status(self) -> Optional[dict]:
        """Get current quota status information."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.get_quota_status(self.quota_limit)
        return None

    def wait_for_quota_if_needed(self) -> None:
        """Wait until quota allows for the next request."""
        if not self.can_make_request():
            import time
            status = self.get_quota_status()
            wait_seconds = status['seconds_until_reset']

            print(f"⏳ [BLS] Quota limit reached ({status['requests_made']}/{self.quota_limit} requests this hour)")
            print(f"⏰ [BLS] Waiting {wait_seconds} seconds until next hour...")

            # Wait with progress updates every 60 seconds
            while wait_seconds > 0:
                if wait_seconds >= 60:
                    print(f"⏰ [BLS] {wait_seconds} seconds remaining until quota reset...")
                    time.sleep(60)
                    wait_seconds -= 60
                else:
                    time.sleep(wait_seconds)
                    wait_seconds = 0

            print("✓ [BLS] Quota reset - resuming crawling")


class FederalFinanceAPI(ApiSource):
    """Federal Finance API data source for T-Bill rates from fiscaldata.treasury.gov, using stock-like data schema."""

    def __init__(self, quota_limit: Optional[int] = None, **config):
        """
        Initialize Federal Finance data source for T-Bill rates.

        Args:
            quota_limit (int, optional): Hourly quota limit for Treasury API
            **config: Additional configuration options
        """
        self.base_url = config.get('base_url', 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service')
        self.quota_limit = quota_limit

        # Initialize quota manager if quota limit is specified
        if quota_limit is not None:
            quota_file = config.get('quota_file', "data/.federal_finance_quota.json")
            self.quota_manager = QuotaManager(quota_file=quota_file)
        else:
            self.quota_manager = None

        # Define T-Bill rate dataset
        self.datasets = {
            'tbill_rates': {
                'endpoint': 'v2/accounting/od/avg_interest_rates',
                'description': 'Treasury Bill Interest Rates',
                'date_field': 'record_date',
                'value_fields': ['avg_interest_rate_amt']
            }
        }

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate symbol (dataset name) for Federal Finance data.

        Args:
            symbol (str): Dataset name (e.g., 'treasury_rates', 'tips_cpi', 'i_bonds')

        Returns:
            bool: True if valid dataset name
        """
        return symbol.lower() in self.datasets

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch federal finance data from Treasury API.

        Args:
            symbol (str): Dataset name (e.g., 'treasury_rates', 'tips_cpi', 'i_bonds')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Federal finance data with standardized columns
        """
        if not self.validate_symbol(symbol):
            available = ', '.join(self.datasets.keys())
            raise ValueError(f"Invalid dataset: {symbol}. Available datasets: {available}")

        # Check quota before making request
        self.wait_for_quota_if_needed()

        dataset = self.datasets[symbol.lower()]
        url = f"{self.base_url}/{dataset['endpoint']}"

        # Build parameters for the API request
        params = {
            'format': 'json',
            'filter': f"{dataset['date_field']}:gte:{start_date},{dataset['date_field']}:lte:{end_date},security_desc:eq:Treasury Bills",
            'sort': f"-{dataset['date_field']}",
            'page[size]': '100000'  # Get max records
        }

        try:
            print(f"📡 [Treasury] Fetching {dataset['description']} from {start_date} to {end_date}...")

            response = requests.get(url, params=params, timeout=30)

            # Record the API request for quota tracking
            if self.quota_manager:
                self.quota_manager.record_request()

            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                raise ValueError(f"No data returned for dataset {symbol}")

            # Create DataFrame from API response
            df = pd.DataFrame(data['data'])

            # Convert date column to datetime and set as index
            date_field = dataset['date_field']
            df[date_field] = pd.to_datetime(df[date_field])
            df.set_index(date_field, inplace=True)
            df.index.name = 'date'

            # Convert value fields to numeric
            for field in dataset['value_fields']:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')

            # Sort by date ascending
            df = df.sort_index()

            # Handle duplicate dates - investigate what's causing them
            if df.index.duplicated().any():
                duplicate_count = df.index.duplicated().sum()
                print(f"⚠️  Found {duplicate_count} duplicate dates, investigating...")

                # Show available columns to understand data structure
                print(f"Available columns: {list(df.columns)}")

                # Find a sample duplicate date to examine
                duplicate_dates = df.index[df.index.duplicated(keep=False)]
                if len(duplicate_dates) > 0:
                    sample_date = duplicate_dates[0]
                    sample_rows = df[df.index == sample_date]
                    print(f"Sample duplicate date {sample_date}:")
                    print(f"Number of rows: {len(sample_rows)}")
                    print("Sample rows:")
                    print(sample_rows.head())

                # Check if duplicates vary by any specific columns
                print("\nChecking for differences between duplicate rows...")
                for col in df.columns:
                    if col != dataset['date_field']:  # Skip the date field itself
                        duplicate_values = df[df.index.duplicated(keep=False)][col].nunique()
                        total_duplicates = len(df[df.index.duplicated(keep=False)])
                        if duplicate_values > 1:
                            print(f"Column '{col}' has {duplicate_values} different values among {total_duplicates} duplicate rows")

                # For now, just keep the first occurrence of each date
                df = df.drop_duplicates(keep='first')
                print(f"✓ Deduplicated to {len(df)} unique dates")

            # Rename columns to standardized format for compatibility
            df = self._standardize_columns(df, symbol.lower())

            print(f"✓ [Treasury] Successfully fetched {len(df)} records")
            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Treasury API request failed: {e}")
        except Exception as e:
            raise Exception(f"Error fetching Treasury data: {e}")

    def _standardize_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Standardize column names to match AshareApiSource database schema (OHLCV format).
        Only preserves standardized columns, drops all other API columns.

        Args:
            df (pd.DataFrame): Raw data from API
            dataset_name (str): Name of the dataset

        Returns:
            pd.DataFrame: DataFrame with only standardized columns matching stock data format
        """
        # For T-Bill rates, map to OHLCV format like AshareApiSource
        if dataset_name == 'tbill_rates':
            if 'avg_interest_rate_amt' in df.columns:
                rate_value = df['avg_interest_rate_amt']

                # Create new DataFrame with only standardized columns
                df_std = pd.DataFrame(index=df.index)

                # Map interest rate to OHLCV columns (same value for all since it's a rate)
                df_std['Open'] = rate_value
                df_std['High'] = rate_value
                df_std['Low'] = rate_value
                df_std['Close'] = rate_value
                df_std['Volume'] = 0  # No volume for interest rates

                # Add adjusted columns with same values (required by AshareApiSource schema)
                df_std['Adj_Open'] = rate_value
                df_std['Adj_High'] = rate_value
                df_std['Adj_Low'] = rate_value
                df_std['Adj_Close'] = rate_value
                df_std['Adj_Volume'] = 0
                df_std['Dividend'] = 0.0
                df_std['Split_Factor'] = 1.0

                return df_std

        # If dataset not recognized, return empty DataFrame with same index
        return pd.DataFrame(index=df.index)

    def get_available_datasets(self) -> dict:
        """
        Get information about available datasets.

        Returns:
            dict: Dictionary with dataset names and descriptions
        """
        return {name: info['description'] for name, info in self.datasets.items()}

    def can_make_request(self) -> bool:
        """Check if we can make another API request within quota."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.can_make_request(self.quota_limit)
        return True  # No quota limit set

    def get_quota_status(self) -> Optional[dict]:
        """Get current quota status information."""
        if self.quota_manager and self.quota_limit:
            return self.quota_manager.get_quota_status(self.quota_limit)
        return None

    def wait_for_quota_if_needed(self) -> None:
        """Wait until quota allows for the next request."""
        if not self.can_make_request():
            import time
            status = self.get_quota_status()
            wait_seconds = status['seconds_until_reset']

            print(f"⏳ [Treasury] Quota limit reached ({status['requests_made']}/{self.quota_limit} requests this hour)")
            print(f"⏰ [Treasury] Waiting {wait_seconds} seconds until next hour...")

            # Wait with progress updates every 60 seconds
            while wait_seconds > 0:
                if wait_seconds >= 60:
                    print(f"⏰ [Treasury] {wait_seconds} seconds remaining until quota reset...")
                    time.sleep(60)
                    wait_seconds -= 60
                else:
                    time.sleep(wait_seconds)
                    wait_seconds = 0

            print("✓ [Treasury] Quota reset - resuming crawling")