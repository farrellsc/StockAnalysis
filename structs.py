from dataclasses import dataclass
from pandas import DataFrame
import json
from typing import Dict, List, Optional
from datetime import datetime
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolios")


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
class MockPortfolio:
    name: str
    trade_history: List[Trade]


@dataclass
class Portfolio:
    name: str
    date: str
    compositions: Dict[str, float]

    @classmethod
    def from_file(cls, filename: str):
        content = json.load(open(PORTFOLIO_DIR + "/" + filename + ".json"))
        # Select the value with the biggest time key
        most_recent_time = max(datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in content.keys())
        most_recent_time_str = most_recent_time.strftime("%Y-%m-%d %H:%M:%S")
        most_recent_portfolio = content[most_recent_time_str]
        return cls(name=filename, date=most_recent_time.strftime("%Y-%m-%d"), compositions=most_recent_portfolio)

    def to_file(self):
        current_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.compositions[current_time_formatted] = self.compositions
        json.dump(self.compositions, open(PORTFOLIO_DIR + "/" + self.name + ".json", "w"))