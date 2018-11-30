import math

import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer


_registry = {}


class AdamW(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def init(state, data):
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(data)
        state['exp_avg_sq'] = torch.zeros_like(data)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.optimize_param(group, p)

        return loss

    def optimize_param(self, group, p):
        grad = p.grad.data
        state = self.state[p]
        if len(state) == 0:
            self.init(state, p.data)

        m, v = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        eps, lr = group['lr'], group['eps']

        m.mul_(beta1).add_(1 - beta1, grad)
        v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        beta1_corrected = 1 - beta1 ** state['step']
        beta2_corrected = 1 - beta2 ** state['step']
        denom = v.sqrt().add(eps)

        if group['weight_decay'] != 0:
            step_size = group['lr']
            m.div_(beta1_corrected)
            v.div_(beta2_corrected)
            m.div_(denom).add_(group['weight_decay']*p.data)

        else:
            step_size = group['lr']*math.sqrt(beta1_corrected)/beta2_corrected

        p.data.addcdiv_(-step_size, m, denom)


def get_optimizer(name_or_opt: str, model: nn.Module,
                  config: dict=None) -> Optimizer:

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
