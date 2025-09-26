import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from datetime import datetime


class Frontend:
    """Frontend class for visualizing stock data using matplotlib."""

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the Frontend with visualization settings.

        Args:
            style (str): Matplotlib style to use. Options: 'seaborn-v0_8', 'ggplot', 'dark_background', etc.
            figsize (tuple): Default figure size (width, height) in inches.
        """
        # Set matplotlib style
        try:
            plt.style.use(style)
        except:
            print(f"⚠️  Style '{style}' not available, using default")
            plt.style.use('default')

        self.default_figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Color palette for multiple stocks

        # Set up seaborn for better aesthetics
        sns.set_palette("husl")

        print(f"✓ Frontend initialized with style: {style}")

    def plot_price_comparison(self,
                              dataframes: List[pd.DataFrame],
                              symbols: List[str],
                              price_column: str = 'Close',
                              normalize: bool = False,
                              show_volume: bool = True,
                              title: str = None,
                              save_path: str = None,
                              figsize: Tuple[int, int] = None) -> plt.Figure:
        """
        Plot multiple stocks on the same graph for comparison.

        Args:
            dataframes (List[pd.DataFrame]): List of dataframes from Backend.get_daily_price()
            symbols (List[str]): List of stock symbols corresponding to dataframes
            price_column (str): Column to plot ('Close', 'Open', 'High', 'Low', 'Adj Close')
            normalize (bool): If True, normalize prices to percentage change from first day
            show_volume (bool): If True, show volume subplot
            title (str, optional): Custom title for the plot
            save_path (str, optional): Path to save the plot
            figsize (tuple, optional): Figure size override

        Returns:
            plt.Figure: The matplotlib figure object

        Raises:
            ValueError: If dataframes and symbols lists don't match or are empty
        """
        # Input validation
        if not dataframes or not symbols:
            raise ValueError("Dataframes and symbols lists cannot be empty")

        if len(dataframes) != len(symbols):
            raise ValueError("Number of dataframes must match number of symbols")

        if price_column not in dataframes[0].columns:
            available_cols = list(dataframes[0].columns)
            raise ValueError(f"Column '{price_column}' not found. Available columns: {available_cols}")

        # Set figure size
        fig_size = figsize or self.default_figsize

        # Create subplots
        if show_volume:
            fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=fig_size,
                                                      gridspec_kw={'height_ratios': [3, 1]},
                                                      sharex=True)
        else:
            fig, ax_price = plt.subplots(1, 1, figsize=fig_size)
            ax_volume = None

        print(f"📈 Plotting comparison chart for {len(symbols)} symbols...")

        # Track data for statistics
        all_data = {}

        # Plot each stock
        for i, (df, symbol) in enumerate(zip(dataframes, symbols)):
            if df.empty:
                print(f"⚠️  Skipping {symbol}: No data available")
                continue

            # Get price data
            price_data = df[price_column].dropna()

            if price_data.empty:
                print(f"⚠️  Skipping {symbol}: No valid price data")
                continue

            # Normalize prices if requested
            if normalize:
                # Convert to percentage change from first day
                normalized_data = (price_data / price_data.iloc[0] - 1) * 100
                ax_price.plot(price_data.index, normalized_data,
                              label=f"{symbol}", linewidth=2, color=self.colors[i % len(self.colors)])
                all_data[symbol] = normalized_data
            else:
                ax_price.plot(price_data.index, price_data,
                              label=f"{symbol}", linewidth=2, color=self.colors[i % len(self.colors)])
                all_data[symbol] = price_data

            # Plot volume if requested
            if show_volume and ax_volume is not None and 'Volume' in df.columns:
                volume_data = df['Volume'].dropna()
                if not volume_data.empty:
                    ax_volume.bar(volume_data.index, volume_data,
                                  alpha=0.6, color=self.colors[i % len(self.colors)],
                                  label=f"{symbol}", width=1)

        # Configure price subplot
        if normalize:
            ax_price.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
            ax_price.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            plot_title = title or f"Normalized Price Comparison ({price_column})"
        else:
            ax_price.set_ylabel(f'{price_column} Price ($)', fontsize=12, fontweight='bold')
            plot_title = title or f"Price Comparison ({price_column})"

        ax_price.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
        ax_price.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax_price.grid(True, alpha=0.3)

        # Format x-axis dates
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        date_range_days = (dataframes[0].index.max() - dataframes[0].index.min()).days

        if date_range_days <= 30:
            ax_price.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, date_range_days // 10)))
        elif date_range_days <= 90:
            ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        elif date_range_days <= 365:
            ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            # For very long ranges
            if date_range_days <= 1095:  # 3 years
                ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:
                ax_price.xaxis.set_major_locator(mdates.YearLocator())

        # Configure volume subplot if shown
        if show_volume and ax_volume is not None:
            ax_volume.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax_volume.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax_volume.grid(True, alpha=0.3)
            ax_volume.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

            # Only show legend if multiple stocks
            if len(symbols) > 1:
                ax_volume.legend(loc='upper right', frameon=True)
        else:
            ax_price.set_xlabel('Date', fontsize=12, fontweight='bold')

        # Rotate date labels for better readability
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add statistics text box
        self._add_statistics_box(ax_price, all_data, symbols, normalize)

        # Adjust layout
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✓ Plot saved to: {save_path}")

        # Print summary statistics
        self._print_summary_stats(all_data, symbols, normalize)

        print(f"✓ Successfully created comparison chart for {len(symbols)} symbols")

        return fig

    def _add_statistics_box(self, ax, data_dict: Dict, symbols: List[str], normalize: bool):
        """Add a statistics box to the plot."""
        if not data_dict:
            return

        stats_text = []
        for symbol in symbols:
            if symbol in data_dict:
                data = data_dict[symbol]
                if not data.empty:
                    latest_value = data.iloc[-1]
                    if normalize:
                        stats_text.append(f"{symbol}: {latest_value:+.1f}%")
                    else:
                        stats_text.append(f"{symbol}: ${latest_value:.2f}")

        if stats_text:
            stats_str = " | ".join(stats_text)
            # Add text box with latest values
            ax.text(0.02, 0.98, f"Latest: {stats_str}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _print_summary_stats(self, data_dict: Dict, symbols: List[str], normalize: bool):
        """Print summary statistics to console."""
        print("\n📊 Summary Statistics:")
        print("-" * 50)

        for symbol in symbols:
            if symbol not in data_dict or data_dict[symbol].empty:
                continue

            data = data_dict[symbol]

            if normalize:
                print(f"{symbol:>6}: Latest: {data.iloc[-1]:+6.1f}% | "
                      f"Min: {data.min():+6.1f}% | Max: {data.max():+6.1f}% | "
                      f"Std: {data.std():5.1f}%")
            else:
                print(f"{symbol:>6}: Latest: ${data.iloc[-1]:8.2f} | "
                      f"Min: ${data.min():8.2f} | Max: ${data.max():8.2f} | "
                      f"Avg: ${data.mean():8.2f}")