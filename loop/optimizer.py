from torch import optim
from torch import nn
from torch.optim import Optimizer


_registry = {}


def get_optimizer(name_or_opt: str, model: nn.Module,
                  config: dict=None) -> Optimizer:

    config = config or {}
    if issubclass(name_or_opt, Optimizer):
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
    return cls(model.parameters(), config)

