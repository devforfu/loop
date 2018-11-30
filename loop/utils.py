from pathlib import Path

from .callbacks import History, Logger, CSVLogger, Checkpoint
from .schedule import Scheduler, CosineAnnealingSchedule


def default_callbacks(workdir=None):
    """Returns a list with commonly used callbacks."""

    workdir = Path(workdir) if workdir else Path.cwd()
    return [
        History(),
        Logger(),
        CSVLogger(filename=workdir/'history.csv'),
        Checkpoint(folder=workdir),
        Scheduler(CosineAnnealingSchedule(), mode='batch')
    ]
