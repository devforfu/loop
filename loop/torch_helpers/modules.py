import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .utils import as_sequential, classifier_weights, get_activation_layer
from .utils import classname, get_output_shape
from ..utils import pairs


class AdaptiveConcatPool2d(nn.Module):
    """Applies average and maximal adaptive pooling to the tensor and
    concatenates results into a single tensor.

    The idea is taken from fastai library.
    """
    def __init__(self, size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.max(x), self.avg(x)], 1)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


def linear(ni, no, dropout=None, bn=True, activ='relu'):
    """Convenience function that creates a linear layer instance with
    'batteries included'.

    The list of created layers:
        * (optionally) BatchNorm
        * (optionally) Dropout
        * Linear
        * Activation (ReLU, LeakyReLU, None, or custom)

    """
    layers = [nn.BatchNorm1d(ni)] if bn else []
    layers.append(nn.Linear(ni, no))
    if dropout is not None and dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(get_activation_layer(activ))
    return layers


class TinyNet(nn.Module):
    """Simplistic convolution network.

    Something suitable for tests and simple datasets training but probably nothing more.
    """
    def __init__(self, n_channels=3, n_out=10, activation=None):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, n_out)
        self.activation = activation

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def tiny_classifier(n_channels, n_out):
    """Creates tiny convnet with log softmax on top of it."""

    activation = nn.LogSoftmax(dim=-1)
    net = TinyNet(n_channels, n_out, activation=activation)
    return net


class FineTunedModel(nn.Module):
    """Builds a fine-tuned network using pretrained backbone and custom head."""

    def __init__(self, n_out, top=None, bn=True, dropout=0.5,
                 arch=models.resnet34, init_fn=classifier_weights,
                 activ=None):

        super().__init__()

        model = arch(True)
        seq_model = as_sequential(model)
        backbone, classifier = seq_model[:-2], seq_model[-2:]
        out_shape = get_output_shape(backbone)
        input_size = out_shape[0] * 2


        def create_top(conf):
            ps = list(pairs(conf))
            for i, (ni, no) in enumerate(ps):
                drop = dropout if dropout else None
                if i < len(ps) - 1:
                    drop /= 2
                yield from linear(ni, no, drop, bn, 'leaky_relu')

            yield nn.Linear(conf[-1], n_out)
            if activ is not None:
                yield activ

        top = [512, 256] if not top else top
        top = [input_size] + top

        self.backbone = backbone
        self.bottleneck = nn.Sequential(AdaptiveConcatPool2d(), Flatten())
        self.top = nn.Sequential(*list(create_top(top)))
        self.init(init_fn)
        self.freeze_backbone()

    def forward(self, x):
        return self.top(self.bottleneck(self.backbone(x)))

    def init(self, func):
        if func is not None:
            self.top.apply(func)

    def freeze_backbone(self, freeze=True, bn=True):
        for child in self.backbone.children():
            name = classname(child)
            if not bn and name.find('BatchNorm') != -1:
                continue
            for p in child.parameters():
                p.requires_grad = not freeze
