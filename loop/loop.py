from collections import defaultdict
from typing import List
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Optimizer

from .callbacks import CallbackGroup, default_callbacks
from .config import defaults
from .optimizer import get_optimizer
from .stepper import BaseStepper


class Loop:

    def __init__(self,
                 model: nn.Module,
                 device: torch.device = None,
                 opt: Optimizer = 'adamw',
                 opt_config: dict = None,
                 move_to_device: bool=True,
                 alpha: float = 0.98):

        opt = get_optimizer(opt, model, opt_config)

        self.model = model
        self.device = device or defaults.device
        self.opt = opt
        self.move_to_device = move_to_device
        self.alpha = alpha
        self.callbacks = None
        self.stop = False
        self.loss = None
        self.ended = False

    def run(self,
            phases: List['Phase'],
            stepper: BaseStepper,
            callbacks=None,
            epochs: int=1,
            loss_fn=None):

        cb = CallbackGroup(callbacks)
        cb.set_loop(self)
        cb.training_start()
        stepper.set_loop(self)

        self.callbacks = cb
        self.loss = loss_fn or F.nll_loss

        for epoch in range(epochs):
            if self.stop:
                break
            metrics = {}
            cb.epoch_start(epoch)
            for phase in phases:
                is_training = phase.grad
                for batch in phase.loader:
                    x, y = self._place_and_unwrap_if_needed(batch)
                    phase.batch_num += 1
                    cb.batch_start(epoch, phase)
                    batch_metrics = stepper.step(x, y, grad=is_training)
                    self.update_metrics(phase, batch_metrics)
                    cb.batch_end(epoch, phase)
                metrics.update({
                    f'{phase.name}_{k}': v
                    for k, v in phase.metrics.items()})
            cb.epoch_end(epoch, metrics)
        cb.training_end()
        self.ended = True

    def update_metrics(self, phase, batch_metrics):
        a = self.alpha
        updated = {}
        for name, new_value in batch_metrics.items():
            old_value = phase.rolling_metrics[name]
            avg_value = a*old_value + (1 - a)*new_value
            debias_value = avg_value/(1 - a**phase.batch_num)
            updated[name] = debias_value
            phase.rolling_metrics[name] = avg_value
        phase.metrics = updated

    def _place_and_unwrap_if_needed(self, batch):
        x, *y = batch
        if self.move_to_device:
            x = x.to(self.device)
            y = [tensor.to(self.device) for tensor in y]
        else:
            x, *y = batch
        if len(y) == 1:
            [y] = y
        return x, y


class Phase:
    """
    Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track
    metrics and counters, related to the specific phase, and keep the reference
    to subset of data, used during phase.
    """
    def __init__(self, name: str, loader: DataLoader, grad: bool=True):
        self.name = name
        self.loader = loader
        self.batch_num = 0
        self.rolling_metrics = defaultdict(lambda: 0)
        self.metrics = None
        self.grad = grad

    def __repr__(self):
        if self.metrics is None:
            return f'<Phase: {self.name}, metrics: none>'
        metrics = ', '.join([
            f'{key}={value:2.4f}'
            for key, value in self.metrics.items()])
        return f'<Phase: {self.name}, metrics: {metrics}>'



def train(model, train_data, valid_data, epochs=1, batch_size=1,
          loss_fn=None, num_workers=None, device=None):

    num_workers = num_workers or cpu_count()
    train_dl = DataLoader(train_data, batch_size, shuffle=True,
                          num_workers=num_workers)
    valid_dl = DataLoader(valid_data, batch_size, num_workers=num_workers)
    phases = [Phase('train', train_dl), Phase('valid', valid_dl, grad=False)]
    loop = Loop(model, device)
    stepper = BaseStepper()
    loop.run(phases, stepper, default_callbacks, epochs, loss_fn)
    return loop
