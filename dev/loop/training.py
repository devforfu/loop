# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 02c_training.ipynb

import sys
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from loop import callbacks as C
from loop.config import defaults
from loop.metrics import accuracy
from loop.modules import TinyNet
from loop.phase import Phase
from loop.utils import unwrap_if_single


def default_optimizer(model, **params):
    if 'lr' not in params:
        params['lr'] = 0.001
    return optim.Adam(model.parameters(), **params)


def create_callbacks(cbs, default: bool=True):
    defaults = [C.RollingLoss(), C.History(), C.StreamLogger()] if default else []
    cbs = list(cbs or [])
    cbs += defaults
    return C.Group(cbs)


_err_stream = sys.stderr
def report_error(exc):
    import traceback
    tb = traceback.format_tb(exc.__traceback__)
    _err_stream.write('Error! Training loop was interupted with un-expected exception\n')
    _err_stream.write('--------------------------------------------------------------\n')
    _err_stream.write('\n'.join(tb))
    _err_stream.write('\n' + str(exc) + '\n')
    _err_stream.write('--------------------------------------------------------------\n')
    _err_stream.flush()


_out_stream = sys.stdout
def get_output_stream():
    return _out_stream
def set_output_stream(stream):
    _out_stream = stream


def write_output(message: str):
    _out_stream.write(message)
    _out_stream.flush()


class Loop:
    """A generic training loop implementation.

    The class wraps model, phases, optimizer and callbacks, and provides a couple of
    methods to make the training process launching a bit more convenient.
    """
    def __init__(self, model: nn.Module, cbs: list=None,
                 default_cb: bool=True, opt: 'optimizer'=None,
                 opt_fn: 'callable'=default_optimizer, opt_params: dict=None,
                 device: 'device'=defaults.device, features_key: str='features',
                 targets_key: str='targets', loss_fn: 'callable'=defaults.loss_fn):

        model.to(device)
        if opt is None:
            opt = opt_fn(model, **(opt_params or {}))

        cb = create_callbacks(cbs, default_cb)
        cb.loop = self

        self.model = model
        self.opt = opt
        self.cb = cb
        self.loss_fn = loss_fn
        self.device = device
        self.features_key = features_key
        self.targets_key = targets_key

    def fit_datasets(self, trn_ds: Dataset, val_ds: Dataset,
                     epochs: int=1, batch_size: int=defaults.batch_size):
        """Uses two torch datasets (training and validation) to fit the model."""
        phases = Phase.make_train_valid(
            trn_ds, val_ds, bs=batch_size,
            num_workers=defaults.num_workers)
        self.train(phases, epochs)

    def fit_loaders(self, loaders: OrderedDict, epochs: int=1):
        """Uses dictionary of loaders to create training phases to fit the model."""
        phases = [
            Phase(name=name, loader=loader, grad=(name == 'train'))
            for name, loader in loaders.items()]
        self.train(phases, epochs)

    def train(self, phases: list, epochs: int=1):
        """Uses a list of training phases to fit the model."""
        try:
            self.cb.training_started(phases=phases, epochs=epochs)
            for epoch in range(1, epochs + 1):
                self.train_one_epoch(phases, epoch)
            self.cb.training_ended(phases=phases)

        except TrainingInterrupted as e:
            self.cb.training_ended(phases=phases)
            self.cb.interrupted(exc=e)

        except Exception as e:
            report_error(e)

        finally:
            self.cb.cleanup()

    def train_one_epoch(self, phases: list, curr_epoch: int=1):
        """Performs a single training iteration."""
        cb, model, opt = self.cb, self.model, self.opt

        phases = Phase.as_named_list(phases)

        cb.epoch_started(epoch=curr_epoch)

        for phase in phases:
            n = len(phase.loader)
            cb.phase_started(phase=phase, total_batches=n)
            is_training = phase.grad
            model.train(is_training)

            for batch_no, batch in enumerate(phase.loader):
                phase.batch_index += 1
                cb.batch_started(phase=phase, total_batches=n)
                x, y = to_xy(batch, self.device)

                with torch.set_grad_enabled(is_training):
                    cb.before_forward(x=x, y=y)
                    out = model(x)
                    cb.after_forward(out=out)
                    loss = self.loss_fn(out, y)
                    cb.after_loss(loss=loss, out=out, target=y)

                if is_training:
                    opt.zero_grad() # move into callback and call `before_backward`
                    cb.before_backward(phase=phase, batch_no=batch_no)
                    loss.backward()
                    opt.step()  # move into callback and call `after_backward`
                    cb.after_backward(phase=phase, batch_no=batch_no)

                phase.batch_loss = loss.item()
                cb.batch_ended(phase=phase, output=out, target=y)

            cb.phase_ended(phase=phase)

        cb.epoch_ended(phases=phases, epoch=curr_epoch)


class TrainingInterrupted(Exception):
    """Exception which is raised in case if training loop is interrupted.

    Note that this exception is intended to 'gracefully' stop a loop and can be raised from a
    callback if it 'decides' that the training should be stopped (e.g. early stopping). All
    other types of exceptions are treated as errors and can't guarantee that the loop is in a
    consistent state.
    """
    def __init__(self, context=None):
        self.context = context
    def __str__(self):
        return str(self.context)


def raise_interruption(context):
    raise TrainingInterrupted(context=context)


def to_xy(batch, device, features_key='features', targets_key='targets'):
    """Converts batch object into (x, y) tuple of samples and targets.

    A batch could be one of the following:
        * tuple with two arrays X and y of the same size
        * dictionary with keys `features_key` and `targets_key`

    """
    if isinstance(batch, (tuple, list)):
        return place_and_unwrap(batch, device)
    elif isinstance(batch, (dict, OrderedDict)):
        x = batch[features_key]
        y = batch[targets_key]
        return place_and_unwrap((x, y), device)
    raise NotImplementedError(f'unknown batch type: {type(batch)}')


def place_and_unwrap(batch, device):
    """Places tensors from batch onto proper device and converts targets
    into proper shape depending on number of tensors.
    """
    batch = [t.to(device) for t in batch]
    x, *ys = batch
    return x, unwrap_if_single(ys)
