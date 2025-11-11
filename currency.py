from dataclasses import dataclass
from typing import Optional, Union


CURRENCY_REGISTRY = {}


def register_currency(c):
    CURRENCY_REGISTRY[c.__name__] = c
    return c


@dataclass
class BaseCurrency:
    amount: float
    value: float

    def to_currency(self, dest):
        if dest.__class__ == self.__class__:
            return self
        return CURRENCY_REGISTRY[dest](self.amount * CURRENCY_REGISTRY[self.__class__.__name__].value / CURRENCY_REGISTRY[dest].value)

    def __sign__(self):
        raise NotImplementedError()

    def __str__(self):
        return f"{self.__sign__()}{self.amount}"

    def __repr__(self):
        return f"{self.__sign__()}{self.amount}"

    def __int__(self):
        return int(self.amount)

    def __float__(self):
        return float(self.amount)

    def __gt__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount > other.amount

    def __lt__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount < other.amount

    def __ge__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount >= other.amount

    def __le__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount <= other.amount

    def __eq__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount == other.amount

    def __ne__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, float) or isinstance(other, int):
            other = self.__class__(amount=other)
        return self.to_currency(other.__class__.__name__).amount != other.amount

    def __add__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(amount=self.amount + other)
        else:
            return self.__class__(amount=self.amount+other.to_currency(self.__class__.__name__).amount)

    def __sub__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(amount=self.amount - other)
        else:
            return self.__class__(amount=self.amount-other.to_currency(self.__class__.__name__).amount)

    def __mul__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(amount=self.amount * other)
        else:
            return self.__class__(amount=self.amount*other.to_currency(self.__class__.__name__).amount)

    def __truediv__(self, other: Union["BaseCurrency", int, float]):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(amount=self.amount / other)
        else:
            return self.__class__(amount=self.amount/other.to_currency(self.__class__.__name__).amount)

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