from pathlib import Path
import pytest

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from loop.torch_helpers.transforms import ExpandChannels


@pytest.fixture(scope='session')
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
