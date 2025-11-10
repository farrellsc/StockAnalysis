from dataclasses import dataclass
from typing import Optional


CURRENCY_REGISTRY = {}


def register_currency(c):
    CURRENCY_REGISTRY[c.__name__] = c
    return c


@dataclass
class BaseCurrency:
    amount: float
    value: float

    def to_currency(self, dest):
        return CURRENCY_REGISTRY[dest](self.amount * CURRENCY_REGISTRY[self.__class__.__name__].value / CURRENCY_REGISTRY[dest].value)

    def __sign__(self):
        raise NotImplementedError()

    def __str__(self):
        return f"{self.__sign__()}{self.amount}"

    def __repr__(self):
        return f"{self.__sign__()}{self.amount}"

@dataclass
@register_currency
class USD(BaseCurrency):
    value = 1

    def __init__(self, amount: float, value: Optional[float] = None):
        self.amount = amount
        self.value = value if value else 1

    def __sign__(self):
        return "$"

    def __repr__(self):
        return f"{self.__sign__()}{self.amount}"


@dataclass
@register_currency
class CNY(BaseCurrency):
    value = 0.14

    def __init__(self, amount: float, value: Optional[float] = None):
        self.amount = amount
        self.value = value if value else 0.14

    def __sign__(self):
        return "¥"

    def __repr__(self):
        return f"{self.__sign__()}{self.amount}"