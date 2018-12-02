from .base import Callback
from ..schedule import ParameterUpdater


class Scheduler(Callback):
    default = [{'name': 'lr'}]

    def __init__(self, schedule, mode='epoch', params_conf=None):
        self.schedule = schedule
        self.params_conf = params_conf or self.default
        self.mode = mode
        self.history = []

    def training_started(self, optimizer, **kwargs):
        self.updater = ParameterUpdater(self.schedule, self.params_conf, optimizer)
        self.updater.save_start_values()

    def batch_ended(self, phase, **kwargs):
        if self.mode == 'batch' and phase.grad:
            self.update_parameters()

    def epoch_ended(self, epoch, **kwargs):
        if self.mode == 'epoch':
            self.update_parameters()

    def update_parameters(self):
        self.history.append(self.updater.current_values())
        self.updater.step()
