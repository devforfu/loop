from pathlib import Path
import psutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import Callback
from ..utils import merge_dicts


class History(Callback):

    def training_started(self, **kwargs):
        self.recorded = None
        self.epochs = []

    def epoch_ended(self, epoch, **kwargs):
        self.epochs.append(epoch)

    def training_ended(self, phases, **kwargs):
        epochs = {'epoch': np.array(self.epochs).astype(int)}
        metrics = [epochs] + [phase.metrics_history for phase in phases]
        data = pd.DataFrame(merge_dicts(metrics))
        data.reset_index(inplace=True, drop=True)
        self.recorded = data

    def plot(self, metrics, ax=None):
        self.recorded.plot(x='epoch', y=metrics, ax=ax)


class RollingLoss(Callback):

    def __init__(self, smooth=0.98):
        self.smooth = smooth

    def batch_ended(self, phase, **kwargs):
        prev = phase.rolling_loss
        a = self.smooth
        avg_loss = a * prev + (1 - a) * phase.batch_loss
        debias_loss = avg_loss / (1 - a ** phase.batch_index)
        phase.rolling_loss = avg_loss
        phase.update(debias_loss)

    def epoch_ended(self, phases, **kwargs):
        for phase in phases:
            phase.update_metric('loss', phase.last_loss)


class MemoryUsage(Callback):

    MB = 1024 ** 2
    GB = 1024 ** 3

    def __init__(self, filename='memory.csv'):
        self.filename = filename
        self._stream = None
        self.iter = None

    def training_started(self, **kwargs):
        self.iter = 0
        if self._stream is not None:
            self.close()
        self._stream = Path(self.filename).open('w')
        self._stream.write(
            'index,mem_percent,mem_free,mem_available,mem_used\n')

    def training_ended(self, **kwargs):
        self.close()

    def batch_ended(self, **kwargs):
        self.iter += 1
        mem = psutil.virtual_memory()
        record = [self.iter, mem.percent, mem.free, mem.available, mem.used]
        string = ','.join([str(x) for x in record])
        self._stream.write(string + '\n')
        self._stream.flush()

    def close(self):
        self._stream.flush()
        self._stream.close()
        self._stream = None

    def plot(self, unit=None, **fig_kwargs):
        unit = unit or self.GB
        mem = pd.read_csv(self.filename)
        index = mem.columns.str.startswith('mem')
        mem[mem.columns[index]] /= unit
        f, ax = plt.subplots(2, 1, **fig_kwargs)
        ax1, ax2 = ax.flat
        unit_name = ('GB' if unit == self.GB else
                     'MB' if unit == self.MB else
                     '')
        self.plot_memory_percentage(ax1, mem)
        self.plot_memory_usage(ax2, mem, unit_name)

    @staticmethod
    def plot_memory_percentage(ax, mem):
        mem.plot(x='index', y='mem_percent', ax=ax)
        ax.set_title('Memory usage during training', fontsize=20)
        ax.set_xlabel('Batch Index', fontsize=16)
        ax.set_ylabel('Percentage', fontsize=16)

    @staticmethod
    def plot_memory_usage(ax, mem, y_label):
        mem.plot(x='index', y=['mem_available', 'mem_used'], ax=ax)
        ax.set_xlabel('Batch Index', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
