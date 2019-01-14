import math

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, nll_loss

from .base import Phase
from ..config import defaults
from ..callbacks import CallbacksGroup, Scheduler, RollingLoss, ProgressBar
from ..schedule import LinearRange, AbsoluteUpdater
from ..shortcuts import create_callbacks


def train(model, opt, phases, callbacks_group, epochs=1, device=defaults.device, loss_fn=nll_loss):
    model.to(device)

    cb = callbacks_group

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


def train_classifier(model, opt, data, epochs=1, batch_size=4, callbacks=None,
                     num_workers=0, device=defaults.device, schedule_params=None):
    """
    Convenience wrapper on top of training loop.

    Prepares training and validation phases, and (optionally) creates a set of callbacks helpful
    for training a classification model.
    """
    train_ds, valid_ds = data
    phases = make_phases(train_ds, valid_ds, batch_size=batch_size, num_workers=num_workers)
    if callbacks is None:
        schedule_params = schedule_params or {}
        callbacks = create_callbacks(len(phases[0].loader), classification=True, **schedule_params)
    callbacks_group = CallbacksGroup(callbacks)
    train(model, opt, phases, callbacks_group, epochs, device, cross_entropy)
    phases = {phase.name: phase for phase in phases}
    return {'callbacks': callbacks_group, 'phases': phases, 'device': device}


def find_lr(model, opt, train_ds, min_lr=1e-7, max_lr=1, batch_size=4,
            loss_fn=nll_loss, log_loss=False):
    """
    Returns a curve that reflects the dependency between learning rate and model loss.

    Greatly inspired by fastai library and L. Smith papers.
    """
    loader = DataLoader(train_ds, batch_size=batch_size, num_workers=defaults.n_cpu)
    phase = Phase('lr_finder', loader, grad=True)
    model_state = model.cpu().state_dict()
    opt.param_groups[0]['lr'] = min_lr
    sched = Scheduler(schedule=LinearRange(len(loader), min_lr, max_lr),
                      updater_cls=AbsoluteUpdater, mode='batch')
    group = CallbacksGroup([RollingLoss(), ProgressBar(), sched])
    opt_state = opt.state_dict()
    train(model, opt, [phase], group, epochs=1, device=defaults.device, loss_fn=loss_fn)
    opt.load_state_dict(opt_state)
    model.cpu().load_state_dict(model_state)
    losses = phase.losses
    if log_loss:
        losses = [math.log10(l) for l in losses]
    lrs = sched.parameter_history('lr')
    return lrs, losses
