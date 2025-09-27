import requests
import pandas as pd
import json
import os
from datetime import datetime
from typing import Optional, List
from abc import ABC, abstractmethod


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

    def __init__(self, api_key: Optional[str] = None, quota_limit: Optional[int] = None, **config):
        """
        Initialize Tiingo data source.

        Args:
            api_key (str, optional): Tiingo API key
            quota_limit (int, optional): Hourly quota limit for Tiingo API
            **config: Additional configuration options
        """
        self.api_key = self._get_api_key(api_key)
        self.base_url = config.get('base_url', 'https://api.tiingo.com/tiingo/daily')
        self.quota_limit = quota_limit

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