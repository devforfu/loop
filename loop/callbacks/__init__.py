from .base import Callback, CallbacksGroup
from .loggers import StreamLogger, ProgressBar
from .metrics import Accuracy
from .schedulers import Scheduler
from .trackers import History, RollingLoss, MemoryUsage


def default_callbacks():
    return [RollingLoss(), History(), StreamLogger()]
