import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .base import Phase
from ..config import defaults


def train_classifier(model, opt, phases, callbacks, epochs):
    train(model, opt, phases, callbacks, epochs, defaults.device, cross_entropy)


def train(model, opt, phases, callbacks, epochs, device, loss_fn):
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


def make_phases(train_ds, valid_ds, batch_size=4):
    return [
        Phase('train', DataLoader(train_ds, batch_size, shuffle=True)),
        Phase('valid', DataLoader(valid_ds, int(batch_size*1.5)), grad=False)
    ]


def place_and_unwrap(batch, dev):
    x, *y = batch
    x = x.to(dev)
    y = [tensor.to(dev) for tensor in y]
    if len(y) == 1:
        [y] = y
    return x, y
