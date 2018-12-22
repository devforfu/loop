from pathlib import Path

import pytest
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from loop import train_classifier, make_phases
from loop import callbacks
from loop.schedule import CosineAnnealingSchedule
from loop.config import defaults
from loop.torch_helpers.modules import TinyNet
from loop.torch_helpers.transforms import ExpandChannels


def test_training_model_with_loop(mnist):
    phases = make_phases(*mnist, batch_size=512)
    model = TinyNet().to(defaults.device)
    opt = Adam(model.parameters(), lr=1e-2)
    cb = callbacks.CallbacksGroup([
        callbacks.RollingLoss(),
        callbacks.Accuracy(),
        callbacks.History(),
        callbacks.Scheduler(
            CosineAnnealingSchedule(
                eta_min=0.01,
                t_max=len(phases[0].loader)),
            mode='batch'
        ),
        callbacks.StreamLogger(),
        callbacks.ProgressBar()
    ])

    train_classifier(model, opt, phases, cb, epochs=3)

    assert phases[1].metrics['accuracy'][-1] > 0.95


def get_transforms(norm_stats):
    return Compose([
        ToTensor(),
        ExpandChannels(3),
        Normalize(*norm_stats)
    ])


@pytest.fixture(scope='module')
def mnist():
    root = Path('~/data/mnist').expanduser()
    stats = (0.1307,), (0.3081,)
    train_ds = MNIST(
        root, train=True, download=True,
        transform=Compose([ToTensor(), ExpandChannels(3), Normalize(*stats)]))
    valid_ds = MNIST(
        root, train=False, download=True,
        transform=Compose([ToTensor(), ExpandChannels(3), Normalize(*stats)]))
    return train_ds, valid_ds
