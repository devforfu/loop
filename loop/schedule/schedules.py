import math


class LinearRange:
    """A scheduler that linearly increase learning rate from min to max values."""

    def __init__(self, t=100, min_lr=1e-7, max_lr=1):
        step_size = (max_lr - min_lr)/(t - 1)
        self.value = min_lr
        self.step_size = step_size

    def update(self, **kwargs):
        value = self.value
        self.value += self.step_size
        return value


class CosineAnnealingSchedule:
    """A schedule that returns eta multiplier in range from 0.0 to 1.0."""

    def __init__(self, eta_min=0.0, eta_max=1.0, t_max=100, t_mult=2):
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.t_max = t_max
        self.t_mult = t_mult
        self.iter = 0

    def update(self, **kwargs):
        self.iter += 1

        eta_min, eta_max, t_max = self.eta_min, self.eta_max, self.t_max

        t = self.iter % t_max
        eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / t_max))
        if t == 0:
            self.iter = 0
            self.t_max *= self.t_mult

        return eta


class OneCycleSchedule:

    def __init__(self, t, linear_pct=0.2, eta_max=1.0, eta_min=None,
                 div_factor=100, decay_to_zero=True):

        if eta_min is None:
            eta_min = eta_max / div_factor

        self.t = t
        self.linear_pct = linear_pct
        self.eta_max = eta_max
        self.eta_min = eta_min

        self.t_cosine = int(math.ceil(t * (1 - linear_pct))) + 1
        self.t_linear = int(math.floor(t * linear_pct))

        self.cosine = CosineAnnealingSchedule(
            eta_min=0 if decay_to_zero else eta_min,
            eta_max=eta_max,
            t_max=self.t_cosine, t_mult=1)
        self.linear = lambda x: x * (eta_max - eta_min) / self.t_linear + eta_min

        self.iter = 0

    def update(self, **kwargs):
        self.iter += 1
        if self.iter <= self.t_linear:
            return self.linear(self.iter)
        else:
            return self.cosine.update()

    @staticmethod
    def from_epochs(loader, epochs):
        return OneCycleSchedule(t=len(loader) * epochs)

