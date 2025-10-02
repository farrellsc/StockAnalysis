
import argparse
from datetime import datetime
import pandas as pd
import os

from database import Database
from frontend import Frontend
from backend import Backend
from mock_trade import MockTrade
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
                         title: str = None):
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
    if portfolios:
        # Use MockTrade for more sophisticated portfolio simulation
        print("Using MockTrade for portfolio simulation...")
        for portfolio in portfolios:
            mock_trader = MockTrade(
                backend=backend,
                trade_history=portfolio.trade_history,
                start_date=start_date,
                end_date=end_date,
                name=portfolio.name,
            )
            portfolio_configs.append(mock_trader.mock(as_stock_config=True))

    all_configs = stocks + portfolio_configs

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


    fig = frontend.plot_price_comparison(
        dataframes=dataframes,
        symbols=symbols,
        ylabel="Price",
        price_column=price_column,
        show_volume=show_volume,
        title=plot_title,
        save_path=save_path,
        secondary_dataframes=secondary_dataframes if secondary_dataframes else None,
        secondary_symbols=secondary_symbols if secondary_symbols else None,
        secondary_ylabel=secondary_ylabel,
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
                print(f"{symbol}: No price data available for {ds}")
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
