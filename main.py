
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import inspect
import json

from pandas import DataFrame

from database import Database
from frontend import Frontend
from backend import Backend
from cert import TiingoKey
from typing import List


REGISTRY = {}


def register(f):
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
    REGISTRY[f.__name__] = f
    return wrapper


@register
def plot_prices(symbols: List[str], start_date: str, end_date: str,
                         normalize: bool = False, show_volume: bool = False,
                         price_column: str = 'Close', save_path: str = None,
                         title: str = None):
    """An API to plot curves by reading the database with input symbols. """
    # Initialize backend and frontend
    frontend = Frontend()
    backend = Backend(database=Database(file_path="./data/stock_data.pkl"))

    # Fetch data for multiple stocks
    dataframes = []

    for symbol in symbols:
        df = backend.get_daily_price(symbol, start_date, end_date)
        dataframes.append(df)

    # Create comparison plot
    plot_title = title or f"Stock Price Comparison: {', '.join(symbols)}"

    fig = frontend.plot_price_comparison(
        dataframes=dataframes,
        symbols=symbols,
        price_column=price_column,
        normalize=normalize,
        show_volume=show_volume,
        title=plot_title,
        save_path=save_path
    )

    if not save_path:
        plt.show()

    return dataframes


@register
def plot_prices_simple(dataframes: List[DataFrame], symbols: List[str], start_date: str, end_date: str,
                         normalize: bool = False, show_volume: bool = False,
                         price_column: str = 'Close', save_path: str = None,
                         title: str = None):
    """An API to plot curves based on input dataframes and symbols. This API does not access database"""
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i][start_date:end_date]

    # Initialize backend and frontend
    frontend = Frontend()

    # Create comparison plot
    plot_title = title or f"Stock Price Comparison: {', '.join(symbols)}"

    fig = frontend.plot_price_comparison(
        dataframes=dataframes,
        symbols=symbols,
        price_column=price_column,
        normalize=normalize,
        show_volume=show_volume,
        title=plot_title,
        save_path=save_path
    )

    if not save_path:
        plt.show()

    return dataframes


@register
def plot_prices_projected(symbols: List[str], weights: List[float], start_date: str, end_date: str,
                         show_volume: bool = False,
                         price_column: str = 'Close', save_path: str = None,
                         title: str = None):
    """An API to read symbol prices and combining them with "weights" to generate the projected performance of a proposed portfolio. Note that this API always normalizes prices to make them comparable."""
    if sum(weights) != 1:
        raise ValueError("weights must sum to 1.")
    if len(symbols) != len(weights):
        raise ValueError("symbols must have same length as weights.")

    # Initialize backend and frontend
    frontend = Frontend()
    backend = Backend(database=Database(file_path="./data/stock_data.pkl"))

    # Fetch data for multiple stocks
    dataframes = []

    for symbol in symbols:
        df = backend.get_daily_price(symbol, start_date, end_date, normalize=True)
        dataframes.append(df)

    aggregated_price = dataframes[0][price_column] * weights[0]
    for i in range(1, len(dataframes)):
        aggregated_price = aggregated_price + dataframes[i][price_column] * weights[i]
    aggregated_price = DataFrame(aggregated_price)
    aggregated_symbol = "+".join([f'[{p:.2f}% {s}]'for s, p in zip(symbols, weights)])
    aggregated_price['symbol'] = aggregated_symbol
    dataframes.append(aggregated_price)
    symbols.append(aggregated_symbol)

    # Create comparison plot
    plot_title = title or f"Stock Price Comparison: {', '.join(symbols)}"

    fig = frontend.plot_price_comparison(
        dataframes=dataframes,
        symbols=symbols,
        price_column=price_column,
        normalize=False,
        show_volume=show_volume,
        title=plot_title,
        save_path=save_path
    )

    if not save_path:
        plt.show()

    return dataframes

@register
def buy_recipe(capital: int, percent: float, distribution: dict):
    """Given a set of parameters, compute the buying recipe"""
    buy_amount = capital * percent
    for symbol, symbol_perc in distribution.items():
        print(f"{symbol}: buy {buy_amount * symbol_perc}")


def parse_date(date_str: str) -> str:
    """Parse and validate date string."""
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Call plot_price_comparison function
      python3 main.py plot_price_comparison --symbols AAPL TSLA GOOGL --start 2024-01-01 --end 2024-12-31
    
      # Plot with normalization and volume
      python3 main.py plot_price_comparison --symbols AAPL MSFT --start 2024-01-01 --end 2024-12-31 --normalize --volume
    
      # Save plot to file
      python3 main.py plot_price_comparison --symbols AAPL TSLA --start 2024-01-01 --end 2024-12-31 --save plot.png
            """
        )

    parser.add_argument('function', help='Function name to call (e.g., plot_price_comparison)')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to plot (e.g., AAPL TSLA GOOGL)')
    parser.add_argument('--start', type=parse_date, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=parse_date, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--normalize', action='store_true', help='Normalize prices to percentage change')
    parser.add_argument('--volume', action='store_true', help='Show volume subplot')
    parser.add_argument('--column', default='Close',
                       choices=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                       help='Price column to plot (default: Close)')
    parser.add_argument('--save', help='Save plot to file (e.g., plot.png)')
    parser.add_argument('--title', help='Custom plot title')

    args = parser.parse_args()

    func = REGISTRY.get(args.function, None)
    # Handle function calls
    if func:
        func(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end,
            normalize=args.normalize,
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
