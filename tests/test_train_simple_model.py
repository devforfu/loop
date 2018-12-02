from pathlib import Path

import pytest
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from loop import train_classifier, make_phases
from loop import callbacks
from loop.schedule import CosineAnnealingSchedule
from loop.config import defaults



def test_training_model_with_loop(mnist):
    phases = make_phases(*mnist, batch_size=512)
    model = Net().to(defaults.device)
    opt = Adam(model.parameters(), lr=1e-2)
    cb = callbacks.CallbacksGroup([
        callbacks.History(),
        callbacks.Scheduler(
            CosineAnnealingSchedule(
                eta_min=0.01,
                t_max=len(phases[0].loader)),
            mode='batch'
        ),
        callbacks.RollingLoss(),
        callbacks.Accuracy(),
        callbacks.StreamLogger(),
        callbacks.ProgressBar()
    ])

    train_classifier(model, opt, phases, cb, epochs=3)

    assert phases[1].metrics['accuracy'][-1] > 0.9


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class ExpandChannels:

    def __init__(self, num_of_channels=3):
        self.nc = num_of_channels

    def __call__(self, x):
        return x.expand((self.nc,) + x.shape[1:])
