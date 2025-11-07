
import argparse
from datetime import datetime
import pandas as pd
import os

from database import Database
from frontend import Frontend
from backend import Backend
from mock_trade import MockTrade
from crawler import Crawler
from typing import Dict, List
from utils import INF
from structs import StockConfig, MacroConfig, Trade, MockPortfolio, Portfolio, BASE_DIR, DATA_DIR
from logging_config import setup_logging, set_verbose_mode, set_quiet_mode


REGISTRY = {}


def register(f):
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
    REGISTRY[f.__name__] = f
    return wrapper


@register
def plot_prices(stocks: List[StockConfig], start_date: str, end_date: str,
                         portfolios: List[MockPortfolio] = [],
                         environments: MacroConfig = MacroConfig(),
                         show_volume: bool = False,
                         price_column: str = 'Close', save_path: str = None,
                         title: str = None, benchmark: str = "SPY"):
    """
    An API to plot curves by reading the database with input symbols.
    """
    # Initialize backend and frontend
    frontend = Frontend()
    backend = Backend(database=Database(file_path=os.path.join(DATA_DIR, "stock_data.pkl")))

    # Fetch data for stocks
    for config in stocks:
        symbol = config.symbol
        if config.data is None:
            if symbol.lower() == 'cpi_inflation':
                config.data = backend.get_yoy_cpi_inflation(start_date, end_date)
            elif symbol.lower() == 'tbill_rates':
                config.data = backend.get_interest_rate(start_date, end_date)
            elif symbol.lower() == 'unemployment_rate':
                config.data = backend.get_unemployment_rate(start_date, end_date)
            else:
                config.data = backend.get_daily_price(symbol, start_date, end_date)

    # Process portfolio to create portfolio tracking dataframe
    portfolio_configs = []
    tags = {}  # Collect trade dates for annotations

    if portfolios:
        # Use MockTrade for more sophisticated portfolio simulation
        print("Using MockTrade for portfolio simulation...")
        for portfolio in portfolios:
            # Collect trade dates and create tags
            for i, trade in enumerate(portfolio.trade_history):
                trade_date = trade.date
                tag_text = f"[#{i+1}][{trade.symbol}] {trade.desc}"
                tags[trade_date] = tag_text

            mock_trader = MockTrade(
                backend=backend,
                trade_history=portfolio.trade_history,
                start_date=start_date,
                end_date=end_date,
                name=portfolio.name,
                benchmark_symbol=benchmark,
            )
            portfolio_configs.append(mock_trader.mock(as_stock_config=True))

    # Add benchmark to the plot
    benchmark_configs = []
    if benchmark and benchmark not in [config.symbol for config in stocks]:
        print(f"Adding benchmark {benchmark} to the plot...")
        benchmark_data = backend.get_daily_price(benchmark, start_date, end_date)
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_config = StockConfig(symbol=f"{benchmark} (Benchmark)", data=benchmark_data, normalize=True)
            benchmark_configs.append(benchmark_config)

    all_configs = portfolio_configs + benchmark_configs + stocks

    for i, config in enumerate(all_configs):
        if config.normalize:
            config.data = backend.normalize_data(config.data)
        data_info = f"{len(config.data)} rows" if config.data is not None else "None"
        print(f"  {i}: {config.symbol} - {data_info}")
        if config.data is not None and len(config.data) > 0:
            if price_column in config.data.columns:
                price_range = f"{config.data[price_column].min():.2f} to {config.data[price_column].max():.2f}"
                print(f"      Price range: {price_range}")
            else:
                print(f"      Columns: {list(config.data.columns)}")

    dataframes = [config.data for config in all_configs]
    symbols = [config.symbol for config in all_configs]

    # Create comparison plot
    plot_title = title or f"Stock Price Comparison: {', '.join(symbols)}"

    secondary_symbols = []
    secondary_dataframes = []
    secondary_ylabel = None
    if environments.interest_rate:
        secondary_symbols.append("interest_rate")
        secondary_dataframes.append(backend.get_interest_rate(start_date, end_date))
        secondary_ylabel = "percent"
    if environments.cpi:
        secondary_symbols.append("cpi")
        secondary_dataframes.append(backend.get_yoy_cpi_inflation(start_date, end_date))
        secondary_ylabel = "percent"
    if environments.unemployment_rate:
        secondary_symbols.append("unemployment_rate")
        secondary_dataframes.append(backend.get_unemployment_rate(start_date, end_date))
        secondary_ylabel = "percent"

    # Separate data into appropriate categories
    stock_dataframes = [config.data for config in stocks]
    stock_symbols = [config.symbol for config in stocks]

    portfolio_dataframes = [config.data for config in portfolio_configs] if portfolio_configs else None
    portfolio_symbols = [config.symbol for config in portfolio_configs] if portfolio_configs else None

    benchmark_dataframes = [config.data for config in benchmark_configs] if benchmark_configs else None
    benchmark_symbols = [config.symbol for config in benchmark_configs] if benchmark_configs else None

    fig = frontend.plot_price_comparison(
        dataframes=stock_dataframes,
        symbols=stock_symbols,
        ylabel="Price",
        price_column=price_column,
        show_volume=show_volume,
        title=plot_title,
        save_path=save_path,
        secondary_dataframes=secondary_dataframes if secondary_dataframes else None,
        secondary_symbols=secondary_symbols if secondary_symbols else None,
        secondary_ylabel=secondary_ylabel,
        tags=tags if tags else None,
        portfolio_dataframes=portfolio_dataframes,
        portfolio_symbols=portfolio_symbols,
        benchmark_dataframes=benchmark_dataframes,
        benchmark_symbols=benchmark_symbols,
    )

    return dataframes

@register
def buy_recipe(capital: int, percent: float, distribution: dict, ds: str):
    """
    Given a set of parameters, compute the buying recipe with actual stock volumes.

    Args:
        capital (int): Total available capital
        percent (float): Percentage of capital to invest (0.0 to 1.0)
        distribution (dict): Symbol -> percentage allocation (e.g., {'AAPL': 0.5, 'GOOGL': 0.5})
        ds (str): Date string in 'YYYY-MM-DD' format to get stock prices

    Returns:
        dict: Symbol -> volume mapping with actual share quantities
    """
    # Initialize backend to get stock prices
    backend = Backend(database=Database(file_path=os.path.join(DATA_DIR, "stock_data.pkl")))

    # Round date to next business day if needed
    target_date = pd.to_datetime(ds)
    if target_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
        target_date = target_date + pd.Timedelta(days=7 - target_date.weekday())
        ds_adjusted = target_date.strftime('%Y-%m-%d')
        print(f"Adjusted weekend date {ds} to business day {ds_adjusted}")
        ds = ds_adjusted

    buy_amount = capital * percent
    recipe = {}

    print(f"Buy Recipe for {ds} with ${buy_amount:.2f} (${capital} × {percent:.1%}):")
    print("-" * 50)

    for symbol, symbol_perc in distribution.items():
        allocation_amount = buy_amount * symbol_perc

        try:
            # Get stock price for the specified date
            price_data = backend.get_daily_price(symbol, ds, ds)

            if price_data is not None and len(price_data) > 0:
                stock_price = price_data['Close'].iloc[0]
                volume = int(allocation_amount / stock_price)
                actual_cost = volume * stock_price

                recipe[symbol] = volume

                print(f"{symbol}: {volume} shares @ ${stock_price:.2f} = ${actual_cost:.2f} "
                      f"({symbol_perc:.1%} allocation)")
            else:
                print(f"{symbol}: No price data available for {ds}, using crawler to fetch data...")
                try:
                    crawler = Crawler()
                    crawler.crawl_single(symbol, ds, ds)

                    backend.database.refresh()
                    price_data = backend.get_daily_price(symbol, ds, ds)

                    if price_data is not None and len(price_data) > 0:
                        stock_price = price_data['Close'].iloc[0]
                        volume = int(allocation_amount / stock_price)
                        actual_cost = volume * stock_price

                        recipe[symbol] = volume

                        print(f"{symbol}: {volume} shares @ ${stock_price:.2f} = ${actual_cost:.2f} "
                              f"({symbol_perc:.1%} allocation) [fetched via crawler]")
                    else:
                        print(f"{symbol}: Failed to fetch price data even with crawler")
                        recipe[symbol] = 0
                except Exception as crawler_error:
                    print(f"{symbol}: Crawler failed - {crawler_error}")
                    recipe[symbol] = 0

        except Exception as e:
            print(f"{symbol}: Error getting price data - {e}")
            recipe[symbol] = 0

    # Calculate total actual cost more efficiently and safely
    total_actual_cost = 0.0
    for symbol in recipe:
        if recipe[symbol] > 0:
            try:
                price_data = backend.get_daily_price(symbol, ds, ds)
                if price_data is not None and len(price_data) > 0:
                    stock_price = price_data['Close'].iloc[0]
                    total_actual_cost += recipe[symbol] * stock_price
            except Exception:
                # Skip symbols with price data issues
                continue

    # Ensure remaining cash is never negative (handle rounding errors)
    remaining_cash = max(0.0, buy_amount - total_actual_cost)

    print("-" * 50)
    print(f"Total cost: ${total_actual_cost:.2f} of ${buy_amount:.2f} allocated")
    print(f"Remaining cash: ${remaining_cash:.2f}")

    return recipe


@register
def refresh_portfolio_data(portfolio_name: str = None):
    """
    Read stock symbols from portfolio JSON files and update the database
    with data up to today's date.

    Args:
        portfolio_name: If specified, only refresh data for this portfolio file.
                       If None, refresh all portfolios.
    """
    import glob
    import json
    from datetime import datetime, timedelta

    # Initialize components
    backend = Backend(database=Database(file_path=os.path.join(DATA_DIR, "stock_data.pkl")))

    print("🔄 Refreshing portfolio data...")
    print("=" * 50)

    # Find portfolio JSON files
    if portfolio_name:
        # Refresh specific portfolio only
        portfolio_file_path = os.path.join(DATA_DIR, "portfolios", f"{portfolio_name}.json")
        if os.path.exists(portfolio_file_path):
            portfolio_files = [portfolio_file_path]
            print(f"📁 Refreshing specific portfolio: {portfolio_name}")
        else:
            print(f"❌ Portfolio file not found: {portfolio_name}.json")
            return
    else:
        # Find all portfolio JSON files
        portfolio_files = glob.glob(os.path.join(DATA_DIR, "portfolios", "*.json"))

        if not portfolio_files:
            print("❌ No portfolio files found in", os.path.join(DATA_DIR, "portfolios"))
            return

        print(f"📁 Found {len(portfolio_files)} portfolio file(s)")

    # Extract all unique symbols from portfolios
    all_symbols = set()

    for portfolio_file in portfolio_files:
        current_portfolio_name = os.path.basename(portfolio_file).replace('.json', '')
        print(f"📄 Processing portfolio: {current_portfolio_name}")

        try:
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)

            # Extract symbols from portfolio allocations
            if 'portfolio' in portfolio_data:
                for symbol in portfolio_data['portfolio'].keys():
                    all_symbols.add(symbol.upper())

            # Extract symbols from records
            if 'records' in portfolio_data:
                for record in portfolio_data['records']:
                    for key in record.keys():
                        if key != 'time':
                            all_symbols.add(key.upper())

            print(f"   ✓ Extracted symbols from {portfolio_name}")

        except Exception as e:
            print(f"   ❌ Error processing {portfolio_name}: {e}")
            continue

    if not all_symbols:
        print("❌ No symbols found in portfolio files")
        return

    symbols_list = sorted(list(all_symbols))
    print(f"\n🎯 Found {len(symbols_list)} unique symbols:")
    print(f"   {', '.join(symbols_list)}")

    # Separate symbols by market type
    chinese_symbols = []
    us_symbols = []

    for symbol in symbols_list:
        if symbol.startswith('SH') or symbol.startswith('SZ'):
            chinese_symbols.append(symbol)
        else:
            us_symbols.append(symbol)

    print(f"\n📊 Symbol breakdown:")
    print(f"   Chinese stocks (SH/SZ): {len(chinese_symbols)} - {chinese_symbols}")
    print(f"   US/Other stocks: {len(us_symbols)} - {us_symbols}")

    # Determine date range for updates
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"📅 Target end date: {today}")

    # Check if today is a weekend and adjust if needed
    today_dt = datetime.now()
    if today_dt.weekday() >= 5:  # Saturday=5, Sunday=6
        days_back = today_dt.weekday() - 4  # Go back to Friday
        adjusted_date = today_dt - timedelta(days=days_back)
        today = adjusted_date.strftime("%Y-%m-%d")
        print(f"📅 Adjusted weekend date to: {today}")

    # Check existing data for all symbols
    print(f"\n📊 Checking database for existing data...")

    chinese_symbols_to_update = []
    us_symbols_to_update = []

    for symbol in symbols_list:
        try:
            date_range = backend.database.get_date_range(symbol=symbol)
            if date_range and date_range.get('end_date'):
                last_date = date_range['end_date']
                print(f"   {symbol}: Latest data until {last_date}")
            else:
                print(f"   {symbol}: No existing data - will fetch full history")

            # Add to appropriate update list
            if symbol.startswith('SH') or symbol.startswith('SZ'):
                chinese_symbols_to_update.append(symbol)
            else:
                us_symbols_to_update.append(symbol)

        except Exception as e:
            print(f"   {symbol}: Error checking data - {e}")
            if symbol.startswith('SH') or symbol.startswith('SZ'):
                chinese_symbols_to_update.append(symbol)
            else:
                us_symbols_to_update.append(symbol)

    if not chinese_symbols_to_update and not us_symbols_to_update:
        print("✅ All symbols are up to date!")
        return

    # Update Chinese stocks using AshareApiSource
    if chinese_symbols_to_update:
        print(f"\n🇨🇳 Updating {len(chinese_symbols_to_update)} Chinese stocks using AshareApiSource...")

        # Calculate fallback start date as smallest existing date for Chinese symbols
        chinese_fallback_start = None
        for symbol in chinese_symbols_to_update:
            try:
                earliest_date = backend.database.get_earliest_date(symbol)
                if earliest_date:
                    if chinese_fallback_start is None or earliest_date < chinese_fallback_start:
                        chinese_fallback_start = earliest_date
            except Exception:
                pass

        # Use 30 days ago as fallback if no existing data found
        if chinese_fallback_start is None:
            from datetime import timedelta
            chinese_fallback_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"📅 Using start date: {chinese_fallback_start}")

        try:
            from api_source import AshareApiSource
            chinese_crawler = Crawler(api_source=AshareApiSource())

            result_path = chinese_crawler.crawl(
                symbols=chinese_symbols_to_update,
                start_date=chinese_fallback_start,
                end_date=today,
                force=False
            )

            print(f"✅ Chinese stocks updated successfully!")

            # Show quota status
            quota_status = chinese_crawler.get_quota_status()
            if quota_status:
                print(f"📈 Ashare API Quota Status: {quota_status}")

        except Exception as e:
            print(f"❌ Error updating Chinese stocks: {e}")

    # Update US/Other stocks using TiingoApiSource (default)
    if us_symbols_to_update:
        print(f"\n🇺🇸 Updating {len(us_symbols_to_update)} US/Other stocks using TiingoApiSource...")

        # Calculate fallback start date as smallest existing date for US symbols
        us_fallback_start = None
        for symbol in us_symbols_to_update:
            try:
                earliest_date = backend.database.get_earliest_date(symbol)
                if earliest_date:
                    if us_fallback_start is None or earliest_date < us_fallback_start:
                        us_fallback_start = earliest_date
            except Exception:
                pass

        # Use 30 days ago as fallback if no existing data found
        if us_fallback_start is None:
            from datetime import timedelta
            us_fallback_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"📅 Using start date: {us_fallback_start}")

        try:
            from api_source import TiingoApiSource
            us_crawler = Crawler(api_source=TiingoApiSource())

            result_path = us_crawler.crawl(
                symbols=us_symbols_to_update,
                start_date=us_fallback_start,
                end_date=today,
                force=True
            )

            print(f"✅ US/Other stocks updated successfully!")

            # Show quota status
            quota_status = us_crawler.get_quota_status()
            if quota_status:
                print(f"📈 Tiingo API Quota Status: {quota_status}")

        except Exception as e:
            print(f"❌ Error updating US/Other stocks: {e}")

    print(f"\n✅ Portfolio data refresh completed!")
    print(f"📁 Updated data saved to database")


def parse_date(date_str: str) -> str:
    """Parse and validate date string."""
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    # Setup centralized logging (easy control!)
    setup_logging(level='INFO')  # Default INFO level for all components

    parser = argparse.ArgumentParser(
        description="Stock Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Call plot_prices function
      python3 main.py plot_prices --symbols AAPL TSLA GOOGL --start 2024-01-01 --end 2024-12-31

      # Plot with normalization for all stocks and volume
      python3 main.py plot_prices --symbols AAPL MSFT --start 2024-01-01 --end 2024-12-31 --normalize --volume

      # Plot with normalization only for specific stocks
      python3 main.py plot_prices --symbols AAPL MSFT GOOGL --start 2024-01-01 --end 2024-12-31 --normalize AAPL GOOGL --volume

      # Save plot to file
      python3 main.py plot_prices --symbols AAPL TSLA --start 2024-01-01 --end 2024-12-31 --save plot.png
            """
        )

    parser.add_argument('function', help='Function name to call (e.g., plot_price_comparison)')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to plot (e.g., AAPL TSLA GOOGL)')
    parser.add_argument('--start', type=parse_date, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=parse_date, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--normalize', nargs='*', help='Normalize specific stocks (provide stock symbols, or omit for all)')
    parser.add_argument('--volume', action='store_true', help='Show volume subplot')
    parser.add_argument('--column', default='Close',
                       choices=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                       help='Price column to plot (default: Close)')
    parser.add_argument('--save', help='Save plot to file (e.g., plot.png)')
    parser.add_argument('--title', help='Custom plot title')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging (DEBUG level)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet logging (WARNING+ only)')

    args = parser.parse_args()

    # Handle logging level arguments
    if args.verbose:
        set_verbose_mode()
    elif args.quiet:
        set_quiet_mode()

    func = REGISTRY.get(args.function, None)
    # Handle function calls
    if func:
        # Create StockConfig objects
        if args.function == 'plot_prices':
            # Determine which stocks to normalize
            normalize_symbols = set()
            if args.normalize is not None:
                if len(args.normalize) == 0:
                    # --normalize with no arguments means normalize all
                    normalize_symbols = set(args.symbols)
                else:
                    # --normalize AAPL TSLA means normalize only those
                    normalize_symbols = set(args.normalize)

            configs = [StockConfig(symbol=symbol, normalize=(symbol in normalize_symbols))
                      for symbol in args.symbols]

            func(
                stocks=configs,
                start_date=args.start,
                end_date=args.end,
                show_volume=args.volume,
                price_column=args.column,
                save_path=args.save,
                title=args.title
            )
        else:
            func(
                symbols=args.symbols,
                start_date=args.start,
                end_date=args.end,
                show_volume=args.volume,
                price_column=args.column,
                save_path=args.save,
                title=args.title
            )
    else:
        print(f"❌ Unknown function: {args.function}")
        print(f"Available functions: {REGISTRY.keys()}")
        parser.print_help()


if __name__ == "__main__":
    main()
