# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 02b_phase.ipynb

from collections import OrderedDict
from typing import Tuple, Union

from torch.utils.data import Dataset, DataLoader

from loop.config import defaults
from loop.utils import broadcast


class NamedList:
    """A convenience wrapper that allows to iterate over ordered dict values and
    access its elements using numerical indicies.
    """
    def __init__(self, od: OrderedDict):
        self.od = OrderedDict(od)
        self.names = list(self.od)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= len(self):
            raise StopIteration()
        return self[self.index]

    def __len__(self):
        return len(self.od)

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            return self.od[item]
        elif isinstance(item, int):
            return self.od[self.names[item]]
        return TypeError(f'invalid index type: {type(item)}')

    def __str__(self):
        return str(self.od)

    def __repr__(self):
        return str(self)


class Phase:
    """Model training loop phase.

    Each model's training loop iteration could be separated into (at least) two
    phases: training and validation. The instances of this class track metrics and
    counters, related to the specific phase, and keep the reference to subset of
    data, used during phase.
    """
    def __init__(self, name: str, loader: DataLoader, grad: bool=True):
        self.name = name
        self.loader = loader
        self.grad = grad
        self.batch_loss = None
        self.batch_index = 0
        self.rolling_loss = 0
        self.losses = []
        self.metrics = OrderedDict()

    def __len__(self):
        return len(self.loader)

    @property
    def last_loss(self):
        return self.losses[-1] if self.losses else None

    @property
    def last_metrics(self):
        metrics = OrderedDict()
        metrics[f'{self.name}_loss'] = self.last_loss
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values[-1]
        return metrics

    @property
    def metrics_history(self):
        metrics = OrderedDict()
        for name, values in self.metrics.items():
            metrics[f'{self.name}_{name}'] = values
        return metrics

    def get_last_value(self, metric):
        return self.last_metrics[f'{self.name}_{metric}']

    def update(self, loss: float):
        self.losses.append(loss)

    def update_metric(self, name: str, value: 'scalar'):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    @staticmethod
    def make_train_valid(trn_ds: Dataset, val_ds: Dataset,
                         bs: int=defaults.batch_size,
                         num_workers: Union[Tuple, int]=0):
        """Creates two loop's phases, train and valid.

        The phases are thin wrappers on top of data loaders intended to track additional
        information gathered during model's fitting process, like loss, performance
        metrics, etc.
        """
        trn, val = broadcast(num_workers, 2)
        phs = OrderedDict()
        phs['train'] = Phase('train', DataLoader(trn_ds, bs, shuffle=True, num_workers=trn))
        phs['valid'] = Phase('valid', DataLoader(val_ds, bs, num_workers=val), grad=False)
        return NamedList(phs)

    @staticmethod
    def as_named_list(phases: list):
        """Converts list of phases into NamedList."""
        return NamedList(OrderedDict([(phase.name, phase) for phase in phases]))

    def __repr__(self):
        return f'Phase(name={self.name}, last_loss={self.last_loss})'

    def __str__(self):
        return repr(self)
