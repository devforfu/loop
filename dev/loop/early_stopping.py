# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 03b_early_stopping.ipynb

from pathlib import Path

import torch

from loop.callbacks import Callback, Order
from loop.utils import autoformat


class BestMetric(Callback):
    """A callback that memorizes the best value of metric.

    The class is intended to be a base class for other types of metric trackers that
    perform some action when metric stops to improve.
    """

    def __init__(self, phase: str='valid', metric: str='loss', better: 'callable'=min):
        self.phase = phase
        self.metric = metric
        self.better = better

    @property
    def formatted_best(self):
        return f'{self.phase}_{self.metric}={autoformat(self.best_value)}'

    def training_started(self, **kwargs):
        self.best_value = None

    def epoch_started(self, **kwargs):
        self.updated = False

    def phase_ended(self, phase, **kwargs):
        ignore = phase.name != self.phase
        if not ignore:
            self.update_best(phase, **kwargs)
        return ignore

    def update_best(self, phase, **kwargs):
        breakpoint()
        new_value = phase.get_last_value(self.metric)
        if self.best_value is None:
            self.best_value = new_value
        else:
            self.best_value = self.better(self.best_value, new_value)
        self.updated = self.best_value == new_value
        return self.updated


class EarlyStopping(BestMetric):

    order = Order.Tracker(1)

    def __init__(self, patience: int=1, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience

    def training_started(self, **kwargs):
        super().training_started(**kwargs)
        self.trials = 0
        self.running = True

    def phase_ended(self, phase, **kwargs):
        ignore = super().phase_ended(phase=phase, **kwargs)
        if ignore: return
        if self.updated:
            self.trials = 0
        else:
            self.trials += 1
            if self.trials >= self.patience:
                self.running = False

    def epoch_ended(self, phases, epoch, **kwargs):
        super().epoch_ended(phases=phases, epoch=epoch, **kwargs)
        if self.running: return
        from loop.training import TrainingInterrupted
        msg = f'Early stopping at epoch {epoch} with {self.formatted_best}'
        raise TrainingInterrupted(msg)


class ModelSaver(BestMetric):

    order = Order.Tracker(2)

    def __init__(self, mode: str='every', root: Path=Path.cwd(), **kwargs):
        super().__init__(**kwargs)
        assert mode in {'every', 'best'}
        self.root = Path(root)
        self.mode = mode

    def training_started(self, **kwargs):
        super().training_started(**kwargs)
        if not self.root.exists():
            self.root.mkdir(parents=True)
        self.last_saved = None

    def epoch_ended(self, phases, epoch, **kwargs):
        super().epoch_ended(phases=phases, epoch=epoch, **kwargs)
        fname = f'model__{self.formatted_best}__epoch={epoch}.pth'
        if self.mode == 'every' or self.updated:
            path = self.root/fname
            torch.save(self.group.model, path)
            self.last_saved = path

    def load_last_saved_state(self, model=None):
        if self.last_saved is None:
            raise ValueError('nothing was saved during training')
        model = model or self.group.model
        if model is None:
            raise ValueError('no model provided to restore the saved state')
        model.load_state_dict(torch.load(self.last_saved))
