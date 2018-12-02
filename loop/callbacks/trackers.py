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
