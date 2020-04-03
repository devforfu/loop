from enum import auto
from typing import Tuple

import pytest

from loop.config import Config, Key, defaults
from loop.core import AutoName


def test_config_contains_all_given_keys():
    keys, config = make_config('Key', {'A': 1, 'B': 2})

    assert keys.A in config
    assert keys.B in config


def test_config_allows_getting_valid_keys():
    keys, config = make_config('Key', {'A': 1, 'B': 2})

    assert config[keys.A] == 1
    assert config[keys.B] == 2


def test_config_allows_setting_valid_keys():
    keys, config = make_config('Key', {'A': 1, 'B': 2})

    config[keys.A] = config[keys.B] = 0

    assert config[keys.A] == 0
    assert config[keys.B] == 0


def test_config_raises_error_on_getting_invalid_key():
    _, config = make_config('Items', {'First': 1})

    with pytest.raises(KeyError):
        _ = config['Unknown']


def test_config_raises_error_on_setting_invalid_key():
    _, config = make_config('Items', {'First': 1})

    with pytest.raises(KeyError):
        config['Unknown'] = 999


def test_config_all_parameters_has_default_values():
    _ = Config(Key, defaults())



def make_config(name: str, kv: dict) -> Tuple[AutoName, Config]:
    keys = AutoName(name, {k: auto() for k in kv.keys()})
    wrapped = {keys[k]: v for k, v in kv.items()}
    config = Config(keys, options=wrapped)
    return keys, config
