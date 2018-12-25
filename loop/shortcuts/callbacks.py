from ..callbacks import StreamLogger, ProgressBar, History, RollingLoss, Accuracy, Scheduler
from ..schedule import CosineAnnealingSchedule


def create_classification_callbacks(n_train_batches, log_every=1):
    """Creates a list of callbacks helpful during classification model training."""

    schedule = CosineAnnealingSchedule(eta_min=0.01, t_max=n_train_batches)
    callbacks = create_logging_callbacks(log_every)
    callbacks += [
        RollingLoss(),
        Accuracy(),
        Scheduler(schedule=schedule, mode='batch')]
    return callbacks


def create_logging_callbacks(log_every=1):
    """Creates a list of callbacks to log model's performance during training."""

    return [
        StreamLogger(log_every=log_every),
        ProgressBar(),
        History()]
