from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

Activation = Callable
Loss = Callable
MaybeActivation = Optional[Activation]
MaybeList = Optional[List]
Func = Union[Callable, str]
Size = Tuple[int, int]


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    @classmethod
    def wrap(cls, value: str) -> Optional['AutoName']:
        try:
            return cls(value)
        except ValueError:
            return None

    @classmethod
    def has_option(cls, value: str) -> bool:
        try:
            _ = cls.wrap(value)
            return True
        except ValueError:
            return False
