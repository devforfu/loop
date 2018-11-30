from pathlib import Path

import pytest
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from loop import train


def test_training_model_with_loop(mnist):
    loop = train(Net(), *mnist, epochs=5, batch_size=512)

    assert loop is not None


@pytest.fixture(scope='module')
def mnist():
    root = Path('~/data/mnist').expanduser()
    stats = (0.1307,), (0.3081,)
    train_ds = MNIST(
        root, train=True, download=True,
        transform=Compose([ToTensor(), Normalize(*stats)]))
    valid_ds = MNIST(
        root, train=False, download=True,
        transform=Compose([ToTensor(), Normalize(*stats)]))
    return train_ds, valid_ds


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
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