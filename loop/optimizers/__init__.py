from torch import optim
from torch.optim import Optimizer

from .adamw import AdamW


_registry = {}


def get_optimizer(name_or_opt, model, config=None):

    config = config or {}
    if issubclass(type(name_or_opt), Optimizer):
        return name_or_opt
    name = name_or_opt
    if hasattr(optim, name):
        cls = getattr(optim, name)
    elif name in _registry:
        cls = _registry[name]
    else:
        raise ValueError(f'unknown optimizer: {name}')
    if not issubclass(cls, Optimizer):
        raise TypeError(f'not an optimizer: {cls}')
    return cls(model.parameters(), **config)


_registry['AdamW'] = AdamW
