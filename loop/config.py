import json
import os
from pathlib import Path
from types import SimpleNamespace
from multiprocessing import cpu_count

import torch


__all__ = ['env_var', 'defaults']


def env_var(var_name, default=None):
    if var_name not in os.environ and default is None:
        raise RuntimeError(
            f'The environment variable {var_name} is not defined but is required to setup the '
            f'package')
    return os.environ.get(var_name, default)


dot_file = Path(env_var('LOOP_DOT_FILE', '~/.loop/config.json')).expanduser()
if dot_file.exists():
    params = json.load(dot_file.open().read())
else:
    params = {}

device = params.get(
    'device', env_var(
        'LOOP_DEFAULT_DEVICE',
        'cuda:0' if torch.cuda.is_available() else 'cpu'))

n_cpu = params.get('n_cpu', env_var('LOOP_CPU_COUNT', cpu_count()))

defaults = dict(device=device, n_cpu=n_cpu)
defaults.update(**params)
defaults = SimpleNamespace(**defaults)

