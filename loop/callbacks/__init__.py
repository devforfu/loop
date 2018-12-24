from .base import Callback, CallbacksGroup
from .loggers import StreamLogger, ProgressBar
from .metrics import Accuracy
from .schedulers import Scheduler
from .trackers import History, RollingLoss, MemoryUsage
from ..utils import to_snake_case


def default_callbacks():
    return [RollingLoss(), History(), StreamLogger()]


def string_to_callback(name, **params):
    """Returns a callback instance from its name."""

    name = to_snake_case(name)
    if name == 'rolling_loss':
        obj = RollingLoss()