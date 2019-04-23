# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 00c_utils.ipynb

from collections import OrderedDict
import re

import numpy as np


__all__ = ['default', 'merge_dicts', 'to_snake_case', 'pairs', 'classname',
           'to_list', 'autoformat', 'is_scalar', 'broadcast', 'unwrap_if_single',
           'from_torch']


def default(x, fallback=None):
    return x if x is not None else fallback


def merge_dicts(ds):
    """Merges a list of dictionaries into single dictionary.

    The order of dicts in the list affects the values of keys in the
    returned dict.
    """
    merged = OrderedDict()
    for d in ds:
        for k, v in d.items():
            merged[k] = v
    return merged


def to_snake_case(string):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def pairs(seq):
    yield from zip(seq[:-1], seq[1:])


def classname(x):
    return x.__class__.__name__


def to_list(obj):
    """Converts iterable into list or wraps a scalar value with list."""
    if isinstance(obj, str):
        return [obj]
    return list(obj) if hasattr(obj, '__len__') or hasattr(obj, '__next__') else [obj]


def autoformat(v):
    """Tryies to convert value into a string using the best possible representation."""

    return (f'{v:d}' if isinstance(v, (int, np.int16, np.int32, np.int64)) else
            f'{v:.4f}' if isinstance(v, (float, np.float16, np.float32, np.float64)) else
            f'{str(v)}')


def is_scalar(obj):
    return isinstance(obj, (int, float, str, complex))


def broadcast(obj, pad=1):
    """Convenience function to unwrap collections and broadcast scalars."""
    if is_scalar(obj):
        return [obj]*pad
    return obj


def unwrap_if_single(obj):
    """Converts obj collection into a scalar if it contains single element only."""
    return obj[0] if len(obj) == 1 else obj


def from_torch(tensor):
    """Converts torch tensor into Numpy array or scalar."""
    obj = tensor.detach().cpu()
    if not obj.shape:
        return obj.item()
    return obj.numpy()
