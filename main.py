
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import inspect
import json

from pandas import DataFrame

from database import Database
from frontend import Frontend
from backend import Backend
from cert import TiingoKey
from typing import Dict, List, Optional

REGISTRY = {}


def register(f):
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
    REGISTRY[f.__name__] = f
    return wrapper


@dataclass
class StockConfig:
    symbol: str
    normalize: bool = False
    data: Optional[DataFrame] = None
    weights: Optional[Dict[str, float]] = None

@dataclass
class Trade:
    symbol: str
    volume: int  # amount of stocks to buy / sell
    date: str  # datetime to perform the trade, e.g. '2025-01-01
    trade: Optional[str] = None  # buy / sell


@dataclass
class MacroConfig:
    interest_rate: bool = False
    cpi: bool = False
    unemployment_rate: bool = False


@register
def plot_prices(stocks: List[StockConfig], start_date: str, end_date: str,
                         trade_history: List[StockConfig] = [],
                         environments: MacroConfig = MacroConfig(),
                         show_volume: bool = False,
                         price_column: str = 'Close', save_path: str = None,
                         title: str = None):
    """
    An API to plot curves by reading the database with input symbols.
    """
    # Initialize backend and frontend
    frontend = Frontend()
    backend = Backend(database=Database(file_path="./data/stock_data.pkl"))

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
        if config.normalize:
            config.data = backend.normalize_data(config.data)

    # Fetch data for projected stocks
    for config in trade_history:
        symbol = config.symbol
        if config.data is None:
            if symbol.lower() in ('cpi_inflation', 'tbill_rates', 'unemployment_rate'):
                raise ValueError("projected_stocks signals only support stock atm")
            aggregated_data = None
            for source, weight in config.weights.items():
                source_data = backend.get_daily_price(source, start_date, end_date)
                # Create a copy to avoid modifying original data
                weighted_data = source_data.copy()
                # Apply weight to all price columns for consistency
                price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                for col in price_columns:
                    if col in weighted_data.columns:
                        weighted_data[col] *= weight

                if aggregated_data is None:
                    aggregated_data = weighted_data
                else:
                    # Ensure proper alignment by using add with fill_value=0
                    aggregated_data = aggregated_data.add(weighted_data, fill_value=0)

            # Only create final DataFrame outside the loop and add symbol
            if aggregated_data is not None:
                aggregated_data = DataFrame(aggregated_data)
                aggregated_data['symbol'] = config.symbol
            if config.normalize:
                config.data = backend.normalize_data(aggregated_data)
            else:
                config.data = aggregated_data

    all_configs = stocks + projected_stocks
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

    args = parser.parse_args()

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
                configs=configs,
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
