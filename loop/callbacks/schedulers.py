from collections import defaultdict

from .base import Callback
from ..schedule import ParameterUpdater


class Scheduler(Callback):
    default = [{'name': 'lr'}]

    def __init__(self, schedule, mode='epoch', params_conf=None, updater_cls=ParameterUpdater):
        self.schedule = schedule
        self.params_conf = params_conf or self.default
        self.mode = mode
        self.history = []
        self.updater_cls = updater_cls

    def training_started(self, optimizer, **kwargs):
        self.updater = self.updater_cls(self.schedule, self.params_conf, optimizer)
        self.updater.save_start_values()

    def batch_ended(self, phase, **kwargs):
        if self.mode == 'batch' and phase.grad:
            self.update_parameters()

    def epoch_ended(self, epoch, **kwargs):
        if self.mode == 'epoch':
            self.update_parameters()

    def update_parameters(self):
        self.updater.step()
        self.history.append(self.updater.current_values())

    def parameter_history(self, name, *names, group_index=0):
        if not self.history:
            return {}
        curve = defaultdict(list)
        names = [name] + list(names)
        for record in self.history:
            group = record[group_index]
            for name in names:
                if name not in group:
                    raise ValueError(f'no history for parameter \'{name}\'')
                curve[name].append(group[name])
        return dict(curve)
