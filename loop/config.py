from collections import namedtuple
import json
import os
from pathlib import Path

import torch


def env_var(var_name, default=None):
    if var_name not in os.environ and default is None:
        raise RuntimeError(
            f'The environment variable {var_name} is not defined but required '
            f'to setup the package')
    return os.environ[var_name]


dot_file = Path(env_var('LOOP_DOT_FILE', '~/.loop/config.json').expanduser())
params = json.load(dot_file.open().read())

device = params.get(
    'device',
    env_var('LOOP_DEFAULT_DEVICE',
            'cuda:0' if torch.cuda.is_available() else 'cpu'))

defaults = dict(device=device)
defaults.update(**params)
defaults = namedtuple('Defaults', sorted(list(defaults.keys)))(**defaults)
