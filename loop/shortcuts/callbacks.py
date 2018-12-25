from ..callbacks import CallbacksGroup, StreamLogger, ProgressBar, History, RollingLoss, Accuracy
from ..callbacks import Scheduler
from ..schedule import CosineAnnealingSchedule


def create_classification_callbacks(n_train_batches, log_every=1):
    """Creates a list of callbacks helpful during classification model training."""

    return CallbacksGroup([
        History(),
        RollingLoss(),
        Accuracy(),
        Scheduler(
            schedule=CosineAnnealingSchedule(eta_min=0.01, t_max=n_train_batches),
            mode='batch'),
        StreamLogger(log_every=log_every),
        ProgressBar(),
    ])
