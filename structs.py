from dataclasses import dataclass
from pandas import DataFrame
from typing import Dict, List, Optional


@dataclass
class StockConfig:
    symbol: str
    normalize: bool = False
    data: Optional[DataFrame] = None


@dataclass
class MacroConfig:
    interest_rate: bool = False
    cpi: bool = False
    unemployment_rate: bool = False


@dataclass
class Trade:
    symbol: str
    volume: int  # amount of stocks to buy (positive) / sell (negative)
    date: str  # datetime to perform the trade, e.g. '2025-01-01'

@dataclass
class Portfolio:
    name: str
    trade_history: List[Trade]