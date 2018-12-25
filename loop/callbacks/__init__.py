from .base import Callback, CallbacksGroup
from .loggers import StreamLogger, ProgressBar
from .metrics import Accuracy
from .schedulers import Scheduler
from .trackers import History, RollingLoss, MemoryUsage
from ..utils import to_snake_case, classname


CALLBACKS_ORDER = {
    'memory_usage': -1,
    'rolling_loss': 0,
    'accuracy': 1,
    'history': 100,
    'scheduler': 200,
    'stream_logger': 1000,
    'progress_bar': 1001,
}

CALLBACKS_CLASSES = {
    'memory_usage': MemoryUsage,
    'rolling_loss': RollingLoss,
    'accuracy': Accuracy,
    'history': History,
    'scheduler': Scheduler,
    'stream_logger': StreamLogger,
    'progress_bar': ProgressBar
}


def sort_callbacks(callbacks: list) -> list:
    """Sorts callbacks to guarantee their execution order.

    The callbacks are invoked in the order of their appearance in the list of callbacks. However,
    some of them should be called strictly before others. For example, StreamLogger callback
    shouldn't be invoked before performance metrics are gathered.

    For this purpose, an explicit ordering is hard-coded in the form of dictionary. Note that this
    ordering is enforced only when the caller doesn't explicitly provide a CallbackGroup.
    """
    def get_order(obj):
        return CALLBACKS_ORDER[to_snake_case(classname(obj))]

    enumerated = [(get_order(cb), cb) for cb in callbacks]
    return [cb for _, cb in sorted(enumerated, key=lambda pair: pair[0])]


def default_callbacks():
    return [RollingLoss(), History(), StreamLogger()]


def get_callback(name_or_cls, **params):
    """Returns a callback instance from its name or instantiates object if class provided."""

    if isinstance(name_or_cls, str):
        name = to_snake_case(name_or_cls)
        if name not in CALLBACKS_CLASSES:
            keys = list(sorted(CALLBACKS_CLASSES.keys()))
            raise ValueError(f'there is no \'{name}\' callback; available names are: {keys}')
        obj = CALLBACKS_CLASSES[name]
    else:
        obj = name_or_cls()
    obj.set_params(**params)
    return obj




