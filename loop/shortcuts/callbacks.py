from ..callbacks import StreamLogger, ProgressBar, History, RollingLoss, Accuracy, Scheduler
from ..schedule import CosineAnnealingSchedule, OneCycleSchedule


__all__ = ['create_callbacks', 'create_logging_callbacks', 'create_classification_callbacks']


# def create_classification_callbacks(n_train_batches, schedule='cos_anneal',
#                                     log_every=1, scheduled_parameters=None,
#                                     **sched_params):
#     """Creates a list of callbacks helpful during classification model training."""
#
#     if isinstance(schedule, str) and schedule not in ('cos_anneal', 'one_cycle'):
#         schedule = 'cos_anneal'
#
#     if schedule == 'cos_anneal':
#         params = {'eta_min': 0.01, 't_max': n_train_batches, **sched_params}
#         schedule = CosineAnnealingSchedule(**params)
#     elif schedule == 'one_cycle':
#         params = {'t': n_train_batches, **sched_params}
#         schedule = OneCycleSchedule(**params)
#
#     callbacks = create_logging_callbacks(log_every)
#     callbacks += [RollingLoss(), Accuracy(), Scheduler(schedule=schedule, mode='batch')]
#     return callbacks


def create_callbacks(n_train_batches, classification=False, schedule='cos_anneal',
                     scheduling_mode='batch', scheduled_params=None,
                     schedule_config=None, log_every=1):

    callbacks = [RollingLoss()]
    callbacks += create_logging_callbacks(log_every)

    if classification:
        callbacks += create_classification_callbacks()

    if schedule is not None:
        schedule_config = schedule_config or {}

        if isinstance(schedule, str) and schedule not in ('cos_anneal', 'one_cycle'):
            schedule = 'cos_anneal'

        if schedule == 'cos_anneal':
            params = {'eta_min': 0.01, 't_max': n_train_batches, **schedule_config}
            schedule = CosineAnnealingSchedule(**params)

        elif schedule == 'one_cycle':
            params = {'t': n_train_batches, **schedule_config}
            schedule = OneCycleSchedule(**params)

        default = [{'name': 'lr'}, {'name': 'weight_decay', 'inverse': True}]
        params_conf = scheduled_params or default
        scheduler = Scheduler(schedule, mode=scheduling_mode, params_conf=params_conf)
        callbacks += [scheduler]

    return callbacks


def create_classification_callbacks():
    return [Accuracy()]


def create_logging_callbacks(log_every=1):
    """Creates a list of callbacks to log model's performance during training."""

    return [StreamLogger(log_every=log_every), ProgressBar(), History()]
