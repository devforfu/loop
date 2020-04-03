from collections import OrderedDict
from enum import auto
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import torch

from loop.core import AutoName


class Key(AutoName):
    BatchSize: 'Data loader batch size' = auto()
    Device: 'Model training device (CPU or GPU)' = auto()
    Datasets: 'Data location root' = auto()
    LossFunc: 'Loss function' = auto()
    NumWorkers: 'Number of workers in parallel pool' = auto()


KeysType = Union[Callable, Iterable]


class Config:
    def __init__(self, key_enum: KeysType, options: OrderedDict):
        check_every_option_has_default_value(key_enum, options)
        self.key_enum = key_enum
        self.options = options

    def __contains__(self, item: Union[Key, str]):
        return self._wrap(item) in self.options

    def __getitem__(self, item: Union[Key, str]):
        return self.options[self._wrap(item)]

    def __setitem__(self, key: Union[Key, str], value: Any):
        self.options[self._wrap(key)] = value

    def __str__(self):
        options = []
        for key, value in self.options.items():
            options.append(f'{key.name}: {value}')
        return '\n'.join(options)

    def _wrap(self, item: Union[Key, str]):
        try:
            return self.key_enum(item)
        except ValueError:
            raise KeyError(f'unknown configuration option: {item}')


def defaults() -> OrderedDict:
    d = OrderedDict()
    d[Key.BatchSize] = 4
    d[Key.Device] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d[Key.Datasets] = Path('~/data')
    d[Key.LossFunc] = torch.nn.functional.nll_loss
    d[Key.NumWorkers] = cpu_count()
    return d


def check_every_option_has_default_value(keys: KeysType, options: dict):
    for k in keys:
        if k not in options:
            raise ValueError(f'a key without default value is found: {k}')
