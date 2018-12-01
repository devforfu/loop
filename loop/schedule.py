from dataclasses import dataclass
import math

from torchvision.models import resnet34

from .callbacks import Callback


class Scheduler(Callback):

    default_parameters = [
        {'name': 'lr'},
        {'name': 'weight_decay', 'inverse': True}
    ]

    def __init__(self, schedule, mode='epoch', parameters=None):
        self.schedule = schedule
        self.parameters = parameters or self.default_parameters
        self.mode = mode
        self.updater = None
        self.history = []

    def training_start(self):
        self.updater = ParameterUpdater(
            self.schedule, self.parameters, self.loop.opt)
        self.history.append(self.updater.current_values())

    def epoch_end(self, epoch, metrics):
        if self.mode == 'epoch':
            self.update_parameters()

    def batch_end(self, epoch, phase):
        if self.mode == 'batch':
            self.update_parameters()

    def update_parameters(self):
        self.history.append(self.updater.current_values())
        self.updater.step()


class ParameterUpdater:

    def __init__(self, sched, params, opt=None):
        self.sched = sched
        self.params = params
        self.opt = opt

    def set_optimizer(self, opt):
        self.opt = opt

    def current_values(self):
        return [
            {conf['name']: group[conf['name']]
             for conf in self.params}
            for group in self.opt.param_groups]

    def step(self):
        mult = self.sched.update()
        for group in self.opt.param_groups:
            for item in self.params:
                name = item['name']
                conf = item.get('conf')
                if name in group:
                    inverse = conf.get('inverse', False)
                    group[name] *= (1 - mult) if inverse else mult


@dataclass
class CosineAnnealingSchedule:
    """
    The schedule class that returns eta multiplier in range from 0.0 to 1.0.
    """
    eta_min: float = 0.0
    eta_max: float = 1.0
    t_max: int = 100
    t_mult: int = 2
    iter: int = 0

    def update(self, **kwargs):
        self.iter += 1

        eta_min, eta_max, t_max = self.eta_min, self.eta_max, self.t_max

        t = self.iter % t_max
        eta = eta_min + 0.5*(eta_max - eta_min)*(1 + math.cos(math.pi*t/t_max))
        if t == 0:
            self.iter = 0
            self.t_max *= self.t_mult

        return eta
