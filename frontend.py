import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from datetime import datetime
import mplcursors
import logging
from logging_config import LoggingConfig


class Frontend:
    """Frontend class for visualizing stock data using matplotlib."""

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (15, 10), log_level: str = 'INFO'):
        """
        Initialize the Frontend with visualization settings.

        Args:
            style (str): Matplotlib style to use. Options: 'seaborn-v0_8', 'ggplot', 'dark_background', etc.
            figsize (tuple): Default figure size (width, height) in inches.
            log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        # Set up logging using centralized config
        self.logger = LoggingConfig.get_logger('frontend')
        if log_level:
            LoggingConfig.set_level_for_component('frontend', log_level)

        # Set matplotlib style
        try:
            plt.style.use(style)
            self.logger.debug(f"Successfully set matplotlib style: {style}")
        except:
            self.logger.warning(f"Style '{style}' not available, using default")
            plt.style.use('default')

        self.default_figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Color palette for multiple stocks

        # Set up seaborn for better aesthetics
        sns.set_palette("husl")

        self.logger.info(f"Frontend initialized with style: {style}, figsize: {figsize}")

    def create_figure(self, show_volume: bool = True, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]:
        """
        Create a new figure and axes for plotting.

        Args:
            show_volume (bool): If True, create subplot for volume
            figsize (tuple, optional): Figure size override

        Returns:
            Tuple[plt.Figure, plt.Axes, Optional[plt.Axes]]: Figure and axes objects (fig, ax_price, ax_volume)
        """
        fig_size = figsize or self.default_figsize

        if show_volume:
            fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=fig_size,
                                                      gridspec_kw={'height_ratios': [3, 1]},
                                                      sharex=True)
        else:
            fig, ax_price = plt.subplots(1, 1, figsize=fig_size)
            ax_volume = None

        return fig, ax_price, ax_volume

    def plot_price_comparison(self,
                              dataframes: List[pd.DataFrame],
                              symbols: List[str],
                              price_column: str = 'Close',
                              ylabel: str = None,
                              show_volume: bool = True,
                              title: str = None,
                              save_path: str = None,
                              figsize: Tuple[int, int] = None,
                              fig: plt.Figure = None,
                              ax_price: plt.Axes = None,
                              ax_volume: plt.Axes = None,
                              secondary_dataframes: List[pd.DataFrame] = None,
                              secondary_symbols: List[str] = None,
                              secondary_price_column: str = 'Close',
                              secondary_ylabel: str = None) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]:
        """
        Plot multiple stocks on the same graph for comparison.

        Args:
            dataframes (List[pd.DataFrame]): List of dataframes from Backend.get_daily_price()
            symbols (List[str]): List of stock symbols corresponding to dataframes
            price_column (str): Column to plot ('Close', 'Open', 'High', 'Low', 'Adj Close')
            show_volume (bool): If True, show volume subplot
            title (str, optional): Custom title for the plot
            save_path (str, optional): Path to save the plot
            figsize (tuple, optional): Figure size override
            fig (plt.Figure, optional): Existing figure to add to
            ax_price (plt.Axes, optional): Existing price axes to add to
            ax_volume (plt.Axes, optional): Existing volume axes to add to
            ylabel (str, optional): Custom label for left y-axis
            secondary_dataframes (List[pd.DataFrame], optional): Second set of dataframes for right y-axis
            secondary_symbols (List[str], optional): List of symbols for secondary data
            secondary_price_column (str): Column to plot for secondary data
            secondary_ylabel (str, optional): Label for right y-axis

        Returns:
            Tuple[plt.Figure, plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]: Figure and axes objects (fig, ax_price, ax_volume, ax_secondary)

        Raises:
            ValueError: If dataframes and symbols lists don't match or are empty
        """
        # Input validation
        if dataframes is None or not symbols:
            self.logger.error("Dataframes and symbols lists cannot be empty")
            raise ValueError("Dataframes and symbols lists cannot be empty")

        if len(dataframes) != len(symbols):
            self.logger.error(f"Number of dataframes ({len(dataframes)}) must match number of symbols ({len(symbols)})")
            raise ValueError("Number of dataframes must match number of symbols")

        self.logger.info(f"Creating price comparison plot for {len(symbols)} symbols: {symbols}")
        self.logger.debug(f"Plot parameters - column: {price_column}, show_volume: {show_volume}, title: {title}")


        if price_column not in dataframes[0].columns:
            available_cols = list(dataframes[0].columns)
            raise ValueError(f"Column '{price_column}' not found. Available columns: {available_cols}")

        # Validate secondary data if provided
        if secondary_dataframes is not None:
            if secondary_symbols is None:
                raise ValueError("secondary_symbols must be provided when secondary_dataframes is given")
            if len(secondary_dataframes) != len(secondary_symbols):
                raise ValueError("Number of secondary dataframes must match number of secondary symbols")
            if secondary_price_column not in secondary_dataframes[0].columns:
                available_cols = list(secondary_dataframes[0].columns)
                raise ValueError(f"Secondary column '{secondary_price_column}' not found. Available columns: {available_cols}")

        # Use existing figure/axes or create new ones
        if fig is None or ax_price is None:
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
        else:
            # Use existing axes - if show_volume is True but ax_volume is None,
            # we'll just plot on the existing price axis
            if show_volume and ax_volume is None:
                print("⚠️  show_volume=True but no ax_volume provided, skipping volume plot")

        print(f"📈 Plotting comparison chart for {len(symbols)} symbols...")

        # Track data for statistics
        all_data = {}

        # Determine starting color index based on existing lines
        existing_lines = len(ax_price.get_lines())

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

            # Calculate color index for this line
            color_idx = (existing_lines + i) % len(self.colors)

            # Plot price data
            ax_price.plot(price_data.index, price_data,
                          label=f"{symbol}", linewidth=2, color=self.colors[color_idx])
            all_data[symbol] = price_data

            # Plot volume if requested
            if show_volume and ax_volume is not None and 'Volume' in df.columns:
                volume_data = df['Volume'].dropna()
                if not volume_data.empty:
                    ax_volume.bar(volume_data.index, volume_data,
                                  alpha=0.6, color=self.colors[color_idx],
                                  label=f"{symbol}", width=1)

        # Plot secondary data on right y-axis if provided
        ax_secondary = None
        if secondary_dataframes is not None and secondary_symbols is not None:
            # Create secondary y-axis
            ax_secondary = ax_price.twinx()

            # Track secondary data for statistics
            secondary_data = {}

            # Plot each secondary stock
            for i, (df, symbol) in enumerate(zip(secondary_dataframes, secondary_symbols)):
                if df.empty:
                    print(f"⚠️  Skipping secondary {symbol}: No data available")
                    continue

                # Get secondary price data
                price_data = df[secondary_price_column].dropna()

                if price_data.empty:
                    print(f"⚠️  Skipping secondary {symbol}: No valid price data")
                    continue

                # Use different line style for secondary data
                color_idx = (existing_lines + len(symbols) + i) % len(self.colors)
                ax_secondary.plot(price_data.index, price_data,
                              label=f"{symbol} (R)", linewidth=2,
                              linestyle='--', color=self.colors[color_idx])
                secondary_data[symbol] = price_data
                all_data[f"{symbol} (R)"] = price_data

            # Configure secondary y-axis
            if secondary_ylabel:
                ax_secondary.set_ylabel(secondary_ylabel, fontsize=12, fontweight='bold')
            else:
                ax_secondary.set_ylabel(f'{secondary_price_column} (Secondary)', fontsize=12, fontweight='bold')

            # Add secondary legend
            ax_secondary.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Create combined symbols list for statistics and hover functionality
        all_symbols = symbols.copy()
        if secondary_symbols:
            all_symbols.extend([f"{s} (R)" for s in secondary_symbols])

        # Configure price subplot (only set labels/title if this is a new plot)
        if existing_lines == 0:
            y_label = ylabel if ylabel is not None else price_column
            ax_price.set_ylabel(y_label, fontsize=12, fontweight='bold')
            plot_title = title or f"Price Comparison ({price_column})"

            ax_price.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
            ax_price.grid(True, alpha=0.3)

        # Always update legend to include new symbols
        ax_price.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

        # Format x-axis dates (only on first call)
        if existing_lines == 0:
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

        # Configure volume subplot if shown (only on first call)
        if existing_lines == 0 and show_volume and ax_volume is not None:
            ax_volume.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax_volume.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax_volume.grid(True, alpha=0.3)
            ax_volume.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        elif existing_lines == 0 and not show_volume:
            ax_price.set_xlabel('Date', fontsize=12, fontweight='bold')

        # Update volume legend if we have multiple series
        if show_volume and ax_volume is not None:
            existing_volume_legend = ax_volume.get_legend()
            total_volume_series = len([child for child in ax_volume.get_children() if hasattr(child, 'get_label') and child.get_label() and child.get_label() != '_nolegend_'])
            if total_volume_series > 1:
                ax_volume.legend(loc='upper right', frameon=True)

        # Rotate date labels for better readability (only on first call)
        if existing_lines == 0:
            plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add 20% margin on top and bottom of y-axis
        y_min, y_max = ax_price.get_ylim()
        y_range = y_max - y_min
        margin = y_range * 0.20
        ax_price.set_ylim(y_min - margin, y_max + margin)

        # Also adjust secondary axis if it exists
        if ax_secondary is not None:
            y_min_sec, y_max_sec = ax_secondary.get_ylim()
            y_range_sec = y_max_sec - y_min_sec
            margin_sec = y_range_sec * 0.20
            ax_secondary.set_ylim(y_min_sec - margin_sec, y_max_sec + margin_sec)

        # Adjust layout (only on first call)
        if existing_lines == 0:
            plt.tight_layout()

        # Add statistics text box after layout is finalized
        self._add_statistics_box(ax_price, all_data, all_symbols)

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            self.logger.info(f"Plot saved to: {save_path}")

        # Add interactive hover functionality
        self._add_hover_functionality(ax_price, all_data, all_symbols)

        # Add hover functionality for secondary axis if it exists
        if ax_secondary is not None:
            self._add_hover_functionality(ax_secondary, all_data, all_symbols)

        # Print summary statistics
        self._print_summary_stats(all_data, all_symbols)

        total_symbols = len(symbols)
        if secondary_symbols:
            total_symbols += len(secondary_symbols)
            self.logger.info(f"Successfully created comparison chart for {len(symbols)} primary and {len(secondary_symbols)} secondary symbols")
        else:
            self.logger.info(f"Successfully created comparison chart for {len(symbols)} symbols")
        print("💡 Hover over the lines to see detailed information")
        print("💡 Use plt.show() to display the plot when ready")

        return fig, ax_price, ax_volume, ax_secondary

    def show_plot(self, fig: plt.Figure = None):
        """
        Display the plot(s). If no figure is provided, shows all current figures.

        Args:
            fig (plt.Figure, optional): Specific figure to show
        """
        if fig is not None:
            fig.show()
        else:
            plt.show()

    def save_plot(self, fig: plt.Figure, save_path: str, dpi: int = 300):
        """
        Save a plot to file.

        Args:
            fig (plt.Figure): Figure to save
            save_path (str): Path to save the plot
            dpi (int): DPI for saved image
        """
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Plot saved to: {save_path}")

    def _add_statistics_box(self, ax, data_dict: Dict, symbols: List[str]):
        """Add a statistics box to the plot, positioned dynamically to avoid overlap with legend."""
        if not data_dict:
            return

        stats_text = []
        for symbol in symbols:
            if symbol in data_dict:
                data = data_dict[symbol]
                if not data.empty:
                    latest_value = data.iloc[-1]
                    # Check if data appears to be normalized (values around 1 or percentages)
                    if abs(latest_value) < 10 and latest_value != int(latest_value):
                        stats_text.append(f"{symbol}: {latest_value:.3f}")
                    else:
                        stats_text.append(f"{symbol}: {latest_value:.2f}")

        if stats_text:
            stats_str = " | ".join(stats_text)

            # Calculate position based on legend to avoid overlap
            legend = ax.get_legend()
            y_position = 0.98  # Default position

            if legend is not None:
                # Estimate legend height based on number of entries
                # Each legend entry is approximately 0.04 in axes coordinates
                num_legend_entries = len(symbols)
                estimated_legend_height = num_legend_entries * 0.04 + 0.02  # Base height + padding

                # Position the statistics box below the estimated legend position
                y_position = 0.98 - estimated_legend_height - 0.02  # Additional padding

                # Ensure we don't go below a reasonable minimum
                y_position = max(y_position, 0.75)

            # Add text box with latest values
            ax.text(0.02, y_position, f"Latest: {stats_str}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _print_summary_stats(self, data_dict: Dict, symbols: List[str]):
        """Print summary statistics to console."""
        print("\n📊 Summary Statistics:")
        print("-" * 50)

        for symbol in symbols:
            if symbol not in data_dict or data_dict[symbol].empty:
                continue

            data = data_dict[symbol]
            latest_value = data.iloc[-1]

            # Auto-detect format based on data range
            if abs(latest_value) < 10 and latest_value != int(latest_value):
                # Likely normalized data
                print(f"{symbol:>6}: Latest: {latest_value:8.3f} | "
                      f"Min: {data.min():8.3f} | Max: {data.max():8.3f} | "
                      f"Avg: {data.mean():8.3f}")
            else:
                # Likely price data
                print(f"{symbol:>6}: Latest: {latest_value:8.2f} | "
                      f"Min: {data.min():8.2f} | Max: {data.max():8.2f} | "
                      f"Avg: {data.mean():8.2f}")

    def _add_hover_functionality(self, ax, data_dict: Dict, symbols: List[str]):
        """Add interactive hover functionality using mplcursors."""
        if not data_dict:
            return

        # Get all the line objects from the plot
        lines = ax.get_lines()

        # Create cursor for all lines
        cursor = mplcursors.cursor(lines, hover=True)

        def on_add(sel):
            # Get the line that was hovered over
            line = sel.artist

            # Find which symbol this line corresponds to
            symbol = None
            for i, (sym, data) in enumerate(zip(symbols, data_dict.values())):
                if i < len(lines) and lines[i] == line:
                    symbol = sym
                    break

            if symbol is None:
                return

            # Get the x (date) and y (price) values at the cursor position
            x_val = sel.target[0]
            y_val = sel.target[1]

            # Convert matplotlib date number back to datetime
            try:
                date_val = mdates.num2date(x_val).strftime('%Y-%m-%d')
            except:
                date_val = str(x_val)

            # Format the annotation text based on data range
            if abs(y_val) < 10 and y_val != int(y_val):
                annotation_text = f"{symbol}\nDate: {date_val}\nValue: {y_val:.3f}"
            else:
                annotation_text = f"{symbol}\nDate: {date_val}\nValue: {y_val:.2f}"

            sel.annotation.set_text(annotation_text)
            sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5",
                                              facecolor="yellow",
                                              alpha=0.8,
                                              edgecolor="black")
            sel.annotation.set_fontsize(10)

        cursor.connect("add", on_add)

        return cursor