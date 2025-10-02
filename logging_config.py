#!/usr/bin/env python3
"""
Centralized logging configuration for the stock analysis system.
"""

import logging
import os
import sys
from typing import Optional, Dict

class LoggingConfig:
    """Centralized logging configuration manager."""

    # Default logging levels for different components
    DEFAULT_LEVELS = {
        'database': 'INFO',
        'backend': 'INFO',
        'frontend': 'INFO',
        'api_source': 'INFO',
        'mock_trade': 'INFO',
        'crawler': 'INFO',
        'root': 'WARNING'
    }

    @classmethod
    def setup_logging(cls,
                     level: str = 'INFO',
                     format_string: Optional[str] = None,
                     log_file: Optional[str] = None,
                     component_levels: Optional[Dict[str, str]] = None) -> None:
        """
        Set up centralized logging configuration.

        Args:
            level (str): Default logging level for all components
            format_string (str, optional): Custom format string
            log_file (str, optional): Path to log file (logs to console if None)
            component_levels (dict, optional): Custom levels per component
                e.g., {'database': 'DEBUG', 'backend': 'INFO'}
        """
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Set component-specific levels
        levels_to_apply = cls.DEFAULT_LEVELS.copy()
        if component_levels:
            levels_to_apply.update(component_levels)

        # Apply component-specific logging levels
        for component, comp_level in levels_to_apply.items():
            logger = logging.getLogger(component)
            logger.setLevel(getattr(logging, comp_level.upper()))

    @classmethod
    def set_level_for_component(cls, component: str, level: str) -> None:
        """
        Set logging level for a specific component.

        Args:
            component (str): Component name (e.g., 'database', 'backend')
            level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        logger = logging.getLogger(component)
        logger.setLevel(getattr(logging, level.upper()))

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            name (str): Logger name

        Returns:
            logging.Logger: Configured logger
        """
        return logging.getLogger(name)

    @classmethod
    def disable_logging(cls) -> None:
        """Disable all logging."""
        logging.disable(logging.CRITICAL)

    @classmethod
    def enable_logging(cls) -> None:
        """Re-enable logging."""
        logging.disable(logging.NOTSET)

    @classmethod
    def set_quiet_mode(cls) -> None:
        """Set all loggers to WARNING or higher (quiet mode)."""
        for component in cls.DEFAULT_LEVELS.keys():
            cls.set_level_for_component(component, 'WARNING')

    @classmethod
    def set_verbose_mode(cls) -> None:
        """Set all loggers to DEBUG level (verbose mode)."""
        for component in cls.DEFAULT_LEVELS.keys():
            cls.set_level_for_component(component, 'DEBUG')

    @classmethod
    def set_normal_mode(cls) -> None:
        """Set all loggers to INFO level (normal mode)."""
        for component in cls.DEFAULT_LEVELS.keys():
            cls.set_level_for_component(component, 'INFO')


# Convenience functions for easy use
def setup_logging(level: str = 'INFO', **kwargs):
    """Convenience function to set up logging."""
    LoggingConfig.setup_logging(level=level, **kwargs)

def set_quiet_mode():
    """Set quiet logging mode."""
    LoggingConfig.set_quiet_mode()

def set_verbose_mode():
    """Set verbose logging mode."""
    LoggingConfig.set_verbose_mode()

def set_normal_mode():
    """Set normal logging mode."""
    LoggingConfig.set_normal_mode()

def disable_logging():
    """Disable all logging."""
    LoggingConfig.disable_logging()

def enable_logging():
    """Re-enable logging."""
    LoggingConfig.enable_logging()


# Example usage patterns
if __name__ == "__main__":
    # Example 1: Basic setup
    setup_logging(level='INFO')

    # Example 2: Custom component levels
    setup_logging(
        level='WARNING',
        component_levels={
            'database': 'DEBUG',
            'backend': 'INFO'
        }
    )

    # Example 3: Log to file
    setup_logging(
        level='INFO',
        log_file='logs/stock_analysis.log'
    )

    # Example 4: Quick mode changes
    set_verbose_mode()  # All DEBUG
    set_quiet_mode()    # All WARNING+
    set_normal_mode()   # All INFO