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
                              secondary_ylabel: str = None,
                              tags: Dict = None,
                              portfolio_dataframes: List[pd.DataFrame] = None,
                              portfolio_symbols: List[str] = None,
                              benchmark_dataframes: List[pd.DataFrame] = None,
                              benchmark_symbols: List[str] = None) -> Tuple[plt.Figure, plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]:
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
            tags (Dict, optional): Dictionary mapping datetime/date strings to annotation strings
            portfolio_dataframes (List[pd.DataFrame], optional): Portfolio dataframes (plotted with thick orange lines)
            portfolio_symbols (List[str], optional): Portfolio symbol names
            benchmark_dataframes (List[pd.DataFrame], optional): Benchmark dataframes (plotted with solid standard lines)
            benchmark_symbols (List[str], optional): Benchmark symbol names

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

        # Validate portfolio parameters
        if portfolio_dataframes is not None:
            if portfolio_symbols is None:
                raise ValueError("portfolio_symbols must be provided when portfolio_dataframes is given")
            if len(portfolio_dataframes) != len(portfolio_symbols):
                raise ValueError("Number of portfolio dataframes must match number of portfolio symbols")

        # Validate benchmark parameters
        if benchmark_dataframes is not None:
            if benchmark_symbols is None:
                raise ValueError("benchmark_symbols must be provided when benchmark_dataframes is given")
            if len(benchmark_dataframes) != len(benchmark_symbols):
                raise ValueError("Number of benchmark dataframes must match number of benchmark symbols")

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

        # Track data and lines for statistics and legend
        all_data = {}
        all_lines = []  # Track all lines for legend ordering

        # Get available colors excluding orange
        available_colors = self._get_non_orange_colors()

        # Determine if portfolios are present (affects line styling for stocks)
        portfolio_present = portfolio_dataframes is not None and len(portfolio_dataframes) > 0

        # Define plot configurations for each data type
        plot_configs = [
            (portfolio_dataframes, portfolio_symbols, {'color': 'orange', 'linewidth': 2, 'linestyle': '-', 'type': 'portfolio'}),
            (benchmark_dataframes, benchmark_symbols, {'linewidth': 2, 'linestyle': '-', 'type': 'benchmark'}),
            (dataframes, symbols, {
                'linewidth': 1 if portfolio_present else 2,
                'linestyle': '--' if portfolio_present else '-',
                'type': 'stock'
            }),
        ]

        color_offset = 0
        for data_list, symbol_list, style_config in plot_configs:
            if data_list is None or symbol_list is None:
                continue

            for i, (df, symbol) in enumerate(zip(data_list, symbol_list)):
                # Validate data
                price_data = self._validate_and_get_price_data(df, symbol, price_column, style_config['type'])
                if price_data is None:
                    continue

                # Determine color
                if 'color' in style_config:
                    color = style_config['color']
                else:
                    color_idx = (color_offset + i) % len(available_colors)
                    color = available_colors[color_idx]

                # Plot price line
                line = ax_price.plot(price_data.index, price_data,
                              label=symbol,
                              linewidth=style_config['linewidth'],
                              linestyle=style_config['linestyle'],
                              color=color)[0]
                all_lines.append(line)
                all_data[symbol] = price_data

                # Plot volume if requested
                self._plot_volume_if_requested(df, symbol, color, show_volume, ax_volume)

            # Update color offset for next data type (only for non-fixed color types)
            if 'color' not in style_config:
                color_offset += len(data_list)

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
                color_idx = (len(all_lines) + i) % len(self.colors)
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
        all_symbols = list(all_data.keys())
        if secondary_symbols:
            all_symbols.extend([f"{s} (R)" for s in secondary_symbols])

        # Configure price subplot (only set labels/title if this is a new plot)
        existing_lines = len(ax_price.get_lines()) - len(all_lines)  # Lines that were there before our additions
        if existing_lines == 0:
            y_label = ylabel if ylabel is not None else price_column
            ax_price.set_ylabel(y_label, fontsize=12, fontweight='bold')
            plot_title = title or f"Price Comparison ({price_column})"

            ax_price.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
            ax_price.grid(True, alpha=0.3)

        # Create legend (lines are already in correct order: portfolios, benchmarks, stocks)
        if all_lines:
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

        # Add tags/annotations if provided
        if tags:
            self._add_tags_to_plot(ax_price, tags, all_data)

        # Check if data appears to be normalized (values around 1.0)
        is_normalized = False
        if all_data:
            # Sample some data points to determine if this looks like normalized data
            sample_values = []
            for data in all_data.values():
                if not data.empty:
                    sample_values.extend(data.iloc[:10].tolist())  # Sample first 10 values

            if sample_values:
                # Consider it normalized if most values are close to 1.0 (between 0.1 and 10)
                close_to_one = sum(1 for val in sample_values if 0.1 <= val <= 10)
                is_normalized = close_to_one / len(sample_values) > 0.8

        # Add horizontal line at y=1 for normalized data to show profit/loss reference
        if is_normalized:
            ax_price.axhline(y=1, color='red', linestyle='-', alpha=0.9, linewidth=3,
                           label='Break-even (1.0)', zorder=10)
            # Update legend to include the reference line
            ax_price.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

        # Set y-axis range with 0 as lower bound and 20% margin on top
        y_min, y_max = ax_price.get_ylim()
        y_range = y_max - y_min
        margin = y_range * 0.20
        ax_price.set_ylim(0, y_max + margin)

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

    def _get_non_orange_colors(self):
        """Get color palette excluding orange variants."""
        import matplotlib.colors as mcolors
        orange_variants = ['orange', 'darkorange', 'orangered']
        available_colors = []

        for c in self.colors:
            try:
                color_hex = mcolors.to_hex(c)
                orange_hex = mcolors.to_hex('orange')
                if color_hex != orange_hex and c not in orange_variants:
                    available_colors.append(c)
            except:
                if str(c).lower() not in orange_variants:
                    available_colors.append(c)

        return available_colors if available_colors else ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def _validate_and_get_price_data(self, df, symbol, price_column, data_type):
        """Validate dataframe and extract price data."""
        if df.empty:
            print(f"⚠️  Skipping {data_type} {symbol}: No data available")
            return None

        price_data = df[price_column].dropna()
        if price_data.empty:
            print(f"⚠️  Skipping {data_type} {symbol}: No valid price data")
            return None

        return price_data

    def _plot_volume_if_requested(self, df, symbol, color, show_volume, ax_volume):
        """Plot volume bar chart if requested and data available."""
        if show_volume and ax_volume is not None and 'Volume' in df.columns:
            volume_data = df['Volume'].dropna()
            if not volume_data.empty:
                ax_volume.bar(volume_data.index, volume_data,
                              alpha=0.6, color=color, label=symbol, width=1)

    def _add_tags_to_plot(self, ax, tags: Dict, all_data: Dict):
        """Add annotations/tags to the plot at specified dates with smart clustering."""
        if not tags or not all_data:
            return

        # Get y-axis limits to position annotations appropriately
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Get some data to determine reasonable y-positions for annotations
        sample_data = next(iter(all_data.values()))

        # Sort tags by date to process them chronologically
        sorted_tags = sorted(tags.items(), key=lambda x: pd.to_datetime(x[0]))

        # Group tags by proximity (within X days = same cluster)
        tag_clusters = self._group_tags_by_proximity(sorted_tags, cluster_days=7)

        # Calculate cluster positions with vertical separation for distant clusters
        cluster_positions = self._calculate_cluster_positions(tag_clusters, all_data, y_min, y_max, y_range, sample_data)

        # Process each cluster with its assigned position
        for i, cluster in enumerate(tag_clusters):
            cluster_y_position = cluster_positions[i]
            self._position_cluster_tags(ax, cluster, all_data, y_min, y_max, y_range, sample_data, cluster_y_position)

        print(f"📌 Added {len(tags)} tag(s) to the plot in {len(tag_clusters)} cluster(s)")

    def _group_tags_by_proximity(self, sorted_tags, cluster_days=7):
        """Group tags that are close in time into clusters."""
        if not sorted_tags:
            return []

        clusters = []
        current_cluster = [sorted_tags[0]]

        for i in range(1, len(sorted_tags)):
            current_date = pd.to_datetime(sorted_tags[i][0])
            prev_date = pd.to_datetime(sorted_tags[i-1][0])

            # If current tag is within cluster_days of the previous tag, add to current cluster
            if abs((current_date - prev_date).days) <= cluster_days:
                current_cluster.append(sorted_tags[i])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [sorted_tags[i]]

        # Don't forget the last cluster
        clusters.append(current_cluster)

        return clusters

    def _calculate_cluster_positions(self, tag_clusters, all_data, y_min, y_max, y_range, sample_data):
        """Calculate y-positions for each cluster, avoiding curve overlap and spreading distant clusters far apart."""
        if not tag_clusters:
            return []

        font_size = 14
        points_to_data_ratio = y_range / 400
        text_height = (font_size + 20) * points_to_data_ratio

        # Analyze curve density across the y-axis to find clear areas
        curve_density_map = self._analyze_curve_density(all_data, y_min, y_max, y_range)

        # Calculate cluster centers based on their average dates and prices
        cluster_info = []
        for cluster in tag_clusters:
            cluster_dates = [pd.to_datetime(tag[0]) for tag in cluster]
            avg_date = pd.to_datetime(sum([d.value for d in cluster_dates]) / len(cluster_dates))

            # Get natural price-based y-position for this cluster
            valid_prices = []
            for symbol, data in all_data.items():
                if not data.empty:
                    closest_date_idx = data.index.get_indexer([avg_date], method='nearest')[0]
                    if 0 <= closest_date_idx < len(data):
                        closest_price = data.iloc[closest_date_idx]
                        valid_prices.append(closest_price)

            natural_y = sum(valid_prices) / len(valid_prices) if valid_prices else y_min + y_range * 0.5
            cluster_info.append((avg_date, natural_y, len(cluster)))

        # Find optimal positions that avoid curves
        cluster_positions = self._find_curve_avoiding_positions(
            cluster_info, curve_density_map, y_min, y_max, y_range, text_height
        )

        return cluster_positions

    def _analyze_curve_density(self, all_data, y_min, y_max, y_range):
        """Analyze where curves are densest to find clear areas for tags."""
        # Create a density map by dividing y-axis into bins
        num_bins = 20
        bin_height = y_range / num_bins
        density_map = [0] * num_bins

        # Count data points in each bin across all curves
        for symbol, data in all_data.items():
            if not data.empty:
                for value in data.values:
                    if pd.notna(value) and y_min <= value <= y_max:
                        bin_index = min(int((value - y_min) / bin_height), num_bins - 1)
                        density_map[bin_index] += 1

        # Convert to density scores (0 = empty, 1 = very dense)
        max_density = max(density_map) if max(density_map) > 0 else 1
        normalized_density = [density / max_density for density in density_map]

        return normalized_density

    def _find_curve_avoiding_positions(self, cluster_info, curve_density_map, y_min, y_max, y_range, text_height):
        """Find positions for clusters that avoid high-density curve areas."""
        num_clusters = len(cluster_info)
        if num_clusters == 0:
            return []

        num_bins = len(curve_density_map)
        bin_height = y_range / num_bins

        # Find low-density bins (good for placing tags)
        low_density_threshold = 0.3  # Bins with density < 30% are considered "clear"
        clear_bins = [i for i, density in enumerate(curve_density_map) if density < low_density_threshold]

        # If no clear bins, use all bins but prefer the lowest density ones
        if not clear_bins:
            clear_bins = sorted(range(num_bins), key=lambda i: curve_density_map[i])

        # Calculate positions for clusters
        cluster_positions = []

        if num_clusters == 1:
            # Single cluster: use the clearest area
            best_bin = clear_bins[0]
            best_y = y_min + (best_bin + 0.5) * bin_height
            cluster_positions.append(best_y)

        elif num_clusters == 2:
            # Two clusters: use top and bottom clear areas
            top_bins = [b for b in clear_bins if b >= num_bins // 2]
            bottom_bins = [b for b in clear_bins if b < num_bins // 2]

            if top_bins and bottom_bins:
                top_bin = max(top_bins)  # Highest clear bin
                bottom_bin = min(bottom_bins)  # Lowest clear bin
            else:
                # Fallback: use extremes
                top_bin = clear_bins[-1] if clear_bins else num_bins - 1
                bottom_bin = clear_bins[0] if clear_bins else 0

            cluster_positions.append(y_min + (bottom_bin + 0.5) * bin_height)
            cluster_positions.append(y_min + (top_bin + 0.5) * bin_height)

        else:
            # Multiple clusters: distribute across clear areas
            # Sort clear bins and select evenly spaced ones
            if len(clear_bins) >= num_clusters:
                # Use the clearest bins, evenly spaced
                step = len(clear_bins) // num_clusters
                selected_bins = [clear_bins[i * step] for i in range(num_clusters)]
            else:
                # Not enough clear bins, distribute across all available clear bins
                # and fill with less-clear bins
                additional_bins_needed = num_clusters - len(clear_bins)
                all_bins_sorted = sorted(range(num_bins), key=lambda i: curve_density_map[i])
                selected_bins = clear_bins + all_bins_sorted[len(clear_bins):len(clear_bins) + additional_bins_needed]

            # Sort bins to maintain order from bottom to top
            selected_bins.sort()

            # Convert bins to y-positions
            for bin_idx in selected_bins:
                y_pos = y_min + (bin_idx + 0.5) * bin_height
                # Ensure position is within bounds and has enough space for text
                y_pos = max(y_min + text_height, min(y_max - text_height, y_pos))
                cluster_positions.append(y_pos)

        return cluster_positions

    def _position_cluster_tags(self, ax, cluster, all_data, y_min, y_max, y_range, sample_data, cluster_y_position=None):
        """Position all tags in a cluster close to each other."""
        if not cluster:
            return

        font_size = 14
        points_to_data_ratio = y_range / 400
        text_height = (font_size + 20) * points_to_data_ratio

        # Use provided cluster position or calculate natural position
        if cluster_y_position is not None:
            cluster_base_y = cluster_y_position
        else:
            # Find the average date for the cluster
            cluster_dates = [pd.to_datetime(tag[0]) for tag in cluster]
            avg_date = pd.to_datetime(sum([d.value for d in cluster_dates]) / len(cluster_dates))

            # Find base y-position for the cluster based on average price at average date
            valid_prices = []
            for symbol, data in all_data.items():
                if not data.empty:
                    closest_date_idx = data.index.get_indexer([avg_date], method='nearest')[0]
                    if 0 <= closest_date_idx < len(data):
                        closest_price = data.iloc[closest_date_idx]
                        valid_prices.append(closest_price)

            if valid_prices:
                cluster_base_y = sum(valid_prices) / len(valid_prices)
            else:
                cluster_base_y = y_min + y_range * 0.5

        # Group tags by exact same date within the cluster
        date_groups = {}
        for date_key, annotation_text in cluster:
            date_str = pd.to_datetime(date_key).strftime('%Y-%m-%d')
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append((date_key, annotation_text))

        # Stack tags in the cluster vertically with tight spacing
        stack_spacing = text_height * 1.1  # Tight spacing within cluster

        # Calculate positions for each date group
        date_group_positions = []
        total_groups = len(date_groups)
        start_offset = -(total_groups - 1) * stack_spacing / 2

        for i, (date_str, tags_in_date) in enumerate(sorted(date_groups.items())):
            group_base_y = cluster_base_y + start_offset + (i * stack_spacing)

            # For multiple tags on the same date, stack them with sufficient spacing
            if len(tags_in_date) > 1:
                same_date_spacing = text_height * 1.2  # Increased spacing for same-date tags
                same_date_start = -(len(tags_in_date) - 1) * same_date_spacing / 2

                for j, (date_key, annotation_text) in enumerate(tags_in_date):
                    tag_y = group_base_y + same_date_start + (j * same_date_spacing)
                    date_group_positions.append((date_key, annotation_text, tag_y))

                # Debug output for same-date tags
                print(f"📅 Same date ({date_str}): {len(tags_in_date)} tags, spacing={same_date_spacing:.3f}")
            else:
                date_key, annotation_text = tags_in_date[0]
                date_group_positions.append((date_key, annotation_text, group_base_y))

        for date_key, annotation_text, tag_y_position in date_group_positions:
            try:
                tag_date = pd.to_datetime(date_key)

                # Check if the date is within our data range
                if not sample_data.empty:
                    data_start = sample_data.index.min()
                    data_end = sample_data.index.max()

                    if tag_date < data_start or tag_date > data_end:
                        print(f"⚠️  Tag date {tag_date.strftime('%Y-%m-%d')} is outside data range, skipping")
                        continue

                # Keep within bounds
                tag_y_position = max(y_min + text_height/2, min(y_max - text_height/2, tag_y_position))

                # Add vertical line at the tag date (behind curves)
                ax.axvline(x=tag_date, color='orange', linestyle=':', alpha=0.8, linewidth=2, zorder=1)

                # Add annotation text (behind curves)
                ax.annotate(annotation_text,
                           xy=(tag_date, tag_y_position),
                           xytext=(10, 20),  # Offset in points
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.0, edgecolor='orange'),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='orange'),
                           fontsize=14,
                           ha='left',
                           zorder=1)

                self.logger.debug(f"Added clustered tag '{annotation_text}' at {tag_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"⚠️  Error adding clustered tag for date {date_key}: {e}")
                continue

    def _find_non_overlapping_position(self, tag_date, preferred_y, used_positions, y_min, y_max, y_range, text_height):
        """Find a y-position that doesn't overlap with existing annotations."""
        # Conservative estimate for date proximity (adjust based on plot time range)
        text_width_days = pd.Timedelta(days=20)  # More conservative estimate

        # Try different vertical positions starting from preferred
        step_size = text_height * 2.0  # Much larger spacing to avoid overlap with font size 14
        max_attempts = 15  # More attempts to find good position

        # Check if there are nearby tags to stack with
        stacking_distance = pd.Timedelta(days=5)  # Stack tags within 5 days
        nearby_tags = [pos for pos in used_positions
                      if abs((tag_date - pos[0]).total_seconds()) < stacking_distance.total_seconds()]

        if nearby_tags:
            # Stack near the closest existing tag
            closest_tag = min(nearby_tags, key=lambda x: abs((tag_date - x[0]).total_seconds()))
            closest_y = closest_tag[1]

            # Try positions directly above and below the closest tag
            stack_positions = [
                closest_y + text_height * 1.1,  # Just above
                closest_y - text_height * 1.1,  # Just below
                closest_y + text_height * 2.2,  # Further above
                closest_y - text_height * 2.2,  # Further below
            ]
            test_positions = [pos for pos in stack_positions
                            if y_min + text_height/2 <= pos <= y_max - text_height/2]
            test_positions.append(preferred_y)  # Fallback to preferred
        else:
            # Generate alternative positions in a spiral pattern (above and below)
            test_positions = [preferred_y]  # Start with preferred
            for i in range(1, max_attempts):
                # Alternate above and below
                if i % 2 == 1:  # Odd numbers go above
                    test_y = preferred_y + (i // 2 + 1) * step_size
                else:  # Even numbers go below
                    test_y = preferred_y - (i // 2) * step_size

                # Keep within plot bounds
                test_y = max(y_min + text_height/2, min(y_max - text_height/2, test_y))
                test_positions.append(test_y)

        # Test each position for overlap
        for test_y in test_positions:
            # Check if this position overlaps with any existing annotation
            overlap_found = False

            for used_x, used_y, used_w, used_h in used_positions:
                # Check for x-axis overlap (date proximity)
                x_overlap = abs((tag_date - used_x).total_seconds()) < text_width_days.total_seconds()

                # Check for y-axis overlap with buffer
                y_overlap = abs(test_y - used_y) < (text_height + used_h) / 2 + text_height * 0.2

                if x_overlap and y_overlap:
                    overlap_found = True
                    break

            if not overlap_found:
                return test_y

        # If all positions overlap, return the one furthest from existing annotations
        if used_positions:
            # Find position with maximum distance from all existing annotations
            best_y = preferred_y
            max_min_distance = 0

            for test_y in test_positions:
                min_distance = float('inf')
                for used_x, used_y, used_w, used_h in used_positions:
                    # Only consider annotations that are close in time
                    if abs((tag_date - used_x).total_seconds()) < text_width_days.total_seconds():
                        distance = abs(test_y - used_y)
                        min_distance = min(min_distance, distance)

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_y = test_y

            return best_y

        return preferred_y