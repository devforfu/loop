# -----------------------------------------
# THIS FILE WAS AUTOGENERATED! DO NOT EDIT!
# -----------------------------------------
# file to edit: 99_testing.ipynb

from torch.nn import functional as F

from loop import callbacks as C
from loop.config import defaults
from loop.metrics import accuracy
from loop.modules import TinyNet
from loop.training import Loop


def get_mnist(flat=False):
    from torchvision.datasets import MNIST
    from torchvision import transforms as T

    root = defaults.datasets/'mnist'

    mean, std = 0.15, 0.15
    mnist_stats = ([mean]*1, [std]*1)

    if flat:
        def flatten(t): return t.flatten()
        def normalize(t): return (t - mean)/std
        tfs = T.Compose([T.ToTensor(), flatten, normalize])
        trn_ds = MNIST(root, train=True, transform=tfs)
        val_ds = MNIST(root, train=False, transform=tfs)

    else:
        trn_ds = MNIST(root, train=True, transform=T.Compose([
            T.Resize(32),
            T.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ]))
        val_ds = MNIST(root, train=False, transform=T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize(*mnist_stats)
        ]))

    return trn_ds, val_ds


def train_classifier_with_callbacks(model, cbs, n, flat=False, bs=1024):
    loop = Loop(model, cbs=cbs, loss_fn=F.cross_entropy)
    loop.fit_datasets(*get_mnist(flat), epochs=n, batch_size=bs)
    return loop
