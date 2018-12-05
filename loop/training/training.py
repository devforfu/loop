import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

import loop.callbacks as C
from .base import Phase
from ..config import defaults
from ..schedule import OneCycleSchedule


class ClassifierTrainer:

    def __init__(self, model, opt, phases):
        self.model = model
        self.opt = opt
        self.phases = phases
        self.callbacks = []

    def __call__(self, *args, **kwargs):
        self.train(*args, **kwargs)

    def train(self, epochs=1, callbacks=None, pbar=True):
        if callbacks is None:
            sched = OneCycleSchedule.from_epochs(self.phases[0].loader, epochs)
            callbacks = [
                C.RollingLoss(),
                C.Accuracy(),
                C.History(),
                C.Scheduler(sched, mode='batch', params_conf=[
                    {'name': 'lr'},
                    {'name': 'weight_decay', 'inverse': True}
                ]),
                C.StreamLogger()
            ]
            if pbar:
                callbacks.append(C.ProgressBar())
        cb = C.CallbacksGroup(callbacks)
        self.callbacks = cb
        train_classifier(self.model, self.opt, self.phases, cb, epochs)

    def __getitem__(self, item):
        return self.callbacks[item]


def train_classifier(model, opt, phases, callbacks, epochs):
    train(model, opt, phases, callbacks, epochs, defaults.device, cross_entropy)


def train(model, opt, phases, callbacks, epochs, device, loss_fn):
    model.to(device)

    cb = callbacks

    cb.training_started(phases=phases, optimizer=opt)

    for epoch in range(1, epochs + 1):
        cb.epoch_started(epoch=epoch)

        for phase in phases:
            n = len(phase.loader)
            cb.phase_started(phase=phase, total_batches=n)
            is_training = phase.grad
            model.train(is_training)

            for batch in phase.loader:

                phase.batch_index += 1
                cb.batch_started(phase=phase, total_batches=n)
                x, y = place_and_unwrap(batch, device)

                with torch.set_grad_enabled(is_training):
                    cb.before_forward_pass()
                    out = model(x)
                    cb.after_forward_pass()
                    loss = loss_fn(out, y)

                if is_training:
                    opt.zero_grad()
                    cb.before_backward_pass()
                    loss.backward()
                    cb.after_backward_pass()
                    opt.step()

                phase.batch_loss = loss.item()
                cb.batch_ended(phase=phase, output=out, target=y)

            cb.phase_ended(phase=phase)

        cb.epoch_ended(phases=phases, epoch=epoch)

    cb.training_ended(phases=phases)


def make_phases(train_ds, valid_ds, batch_size=4, num_workers=0):
    if isinstance(num_workers, tuple):
        trn, val = num_workers
    else:
        trn = val = num_workers
    return [
        Phase('train', DataLoader(
            train_ds, batch_size, shuffle=True, num_workers=trn)),
        Phase('valid', DataLoader(
            valid_ds, batch_size, num_workers=val), grad=False)]


def place_and_unwrap(batch, dev):
    x, *y = batch
    x = x.to(dev)
    y = [tensor.to(dev) for tensor in y]
    if len(y) == 1:
        [y] = y
    return x, y
