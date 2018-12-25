"""
Base classes to implement callbacks-based loop.

The loop is only responsible for binding data loader and model together,
while the rest of training tricks is implemented as callbacks.
"""
from . import sort_callbacks
from ..utils import to_snake_case, classname
from ..mixins import ParametersMixin


class Callback(ParametersMixin):
    """
    The base class inherited by callbacks.

    Provides a lot of hooks invoked on various stages of the training loop
    execution. The signature of functions is as broad as possible to allow
    flexibility and customization in descendant classes.
    """

    def training_started(self, **kwargs):
        pass

    def training_ended(self, **kwargs):
        pass

    def epoch_started(self, **kwargs):
        pass

    def phase_started(self, **kwargs):
        pass

    def phase_ended(self, **kwargs):
        pass

    def epoch_ended(self, **kwargs):
        pass

    def batch_started(self, **kwargs):
        pass

    def batch_ended(self, **kwargs):
        pass

    def before_forward_pass(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def before_backward_pass(self, **kwargs):
        pass

    def after_backward_pass(self, **kwargs):
        pass


class CallbacksGroup(Callback):

    def __init__(self, callbacks):
        self.callbacks = sort_callbacks(callbacks)
        self.named_callbacks = {to_snake_case(classname(cb)): cb for cb in callbacks}

    def __getitem__(self, item):
        item = to_snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name not found: {item}')

    def training_started(self, **kwargs):
        for cb in self.callbacks:
            cb.training_started(**kwargs)

    def training_ended(self, **kwargs):
        for cb in self.callbacks:
            cb.training_ended(**kwargs)

    def phase_started(self, **kwargs):
        for cb in self.callbacks:
            cb.phase_started(**kwargs)

    def phase_ended(self, **kwargs):
        for cb in self.callbacks:
            cb.phase_ended(**kwargs)

    def epoch_started(self, **kwargs):
        for cb in self.callbacks:
            cb.epoch_started(**kwargs)

    def epoch_ended(self, **kwargs):
        for cb in self.callbacks:
            cb.epoch_ended(**kwargs)

    def batch_started(self, **kwargs):
        for cb in self.callbacks:
            cb.batch_started(**kwargs)

    def batch_ended(self, **kwargs):
        for cb in self.callbacks:
            cb.batch_ended(**kwargs)

    def before_forward_pass(self, **kwargs):
        for cb in self.callbacks:
            cb.before_forward_pass(**kwargs)

    def after_forward_pass(self, **kwargs):
        for cb in self.callbacks:
            cb.after_forward_pass(**kwargs)

    def before_backward_pass(self, **kwargs):
        for cb in self.callbacks:
            cb.before_forward_pass(**kwargs)

    def after_backward_pass(self, **kwargs):
        for cb in self.callbacks:
            cb.after_backward_pass(**kwargs)